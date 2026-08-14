import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from typing_extensions import Annotated

from pocket_tts.data.audio import StreamingWAVWriter, stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    MAX_TOKEN_PER_CHUNK,
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts.models.tts_model import TTSModel, export_model_state
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Saved-Filename"],
)


# ------------------------------------------------------
# Almacenamiento local de audios generados (para que otras apps
# en la misma máquina puedan listarlos/descargarlos por HTTP)
# ------------------------------------------------------

STORAGE_DIR = Path.home() / "Andres Duque Studio - Clonador de Voz" / "audios"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
AUTO_NAME_PREFIX = "adc"


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._áéíóúÁÉÍÓÚñÑ -]", "", name).strip()
    safe = re.sub(r"\s+", "_", safe)
    return safe or "audio"


def _next_auto_name() -> str:
    existing_nums = []
    for f in STORAGE_DIR.glob(f"{AUTO_NAME_PREFIX}_*.wav"):
        m = re.match(rf"{AUTO_NAME_PREFIX}_(\d+)\.wav$", f.name)
        if m:
            existing_nums.append(int(m.group(1)))
    next_n = (max(existing_nums) + 1) if existing_nums else 1
    return f"{AUTO_NAME_PREFIX}_{next_n:03d}"


def _resolve_save_path(save_as: str | None) -> Path:
    base = _sanitize_filename(save_as.strip()) if save_as and save_as.strip() else _next_auto_name()
    path = STORAGE_DIR / f"{base}.wav"
    counter = 1
    while path.exists():
        path = STORAGE_DIR / f"{base}_{counter}.wav"
        counter += 1
    return path


def _safe_storage_path(filename: str) -> Path:
    """Evita path traversal: solo permite nombres de archivo simples dentro de STORAGE_DIR."""
    return STORAGE_DIR / Path(filename).name


# ------------------------------------------------------
# Almacenamiento local de voces guardadas (referencias para clonar) --
# mismo patrón que los audios: compartido por HTTP entre apps locales.
# ------------------------------------------------------

VOICES_DIR = Path.home() / "Andres Duque Studio - Clonador de Voz" / "voices"
VOICES_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_voice_path(name: str) -> Path:
    base = _sanitize_filename(name.strip()) if name and name.strip() else "voz"
    path = VOICES_DIR / f"{base}.wav"
    counter = 1
    while path.exists():
        path = VOICES_DIR / f"{base}_{counter}.wav"
        counter += 1
    return path


def _safe_voice_path(filename: str) -> Path:
    """Evita path traversal: solo permite nombres de archivo simples dentro de VOICES_DIR."""
    return VOICES_DIR / Path(filename).name


def _write_metadata(save_path: Path, text: str | None, label: str | None, kind: str):
    meta = {"text": text, "label": label, "kind": kind, "created_at": time.time()}
    save_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))


def _read_metadata(save_path: Path) -> dict:
    meta_path = save_path.with_suffix(".json")
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


@web_app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    content = static_path.read_text()
    # Replace the placeholder with the actual default text prompt
    print(str(tts_model.origin))
    content = content.replace(
        "DEFAULT_TEXT_PROMPT", get_default_text_for_language(str(tts_model.origin))
    )
    return content


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


def write_to_queue(queue, text_to_generate, model_state, frames_after_eos, save_path):
    """Allows writing to the StreamingResponse as if it were a file.

    Also tees every generated chunk to a real file on disk (`save_path`) using a
    second, separate WAV writer -- the disk copy is seekable so its header ends up
    with the correct final size, unlike the one streamed to the browser.
    """

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)
            return len(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    sample_rate = tts_model.config.mimi.sample_rate
    stream_target = FileLikeToQueue(queue)
    disk_file = open(save_path, "wb") if save_path is not None else None

    stream_writer = StreamingWAVWriter(stream_target, sample_rate)
    stream_writer.write_header(sample_rate)
    disk_writer = None
    if disk_file is not None:
        disk_writer = StreamingWAVWriter(disk_file, sample_rate)
        disk_writer.write_header(sample_rate)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state,
        text_to_generate=text_to_generate,
        frames_after_eos=frames_after_eos,
    )
    for chunk in audio_chunks:
        stream_writer.write_pcm_data(chunk)
        if disk_writer is not None:
            disk_writer.write_pcm_data(chunk)

    stream_writer.finalize()
    if disk_writer is not None:
        disk_writer.finalize()
        disk_file.close()
    stream_target.close()


def generate_data_with_state(
    text_to_generate: str, model_state: dict, frames_after_eos: int, save_path: Path | None
):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(
        target=write_to_queue,
        args=(queue, text_to_generate, model_state, frames_after_eos, save_path),
    )
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    frames_after_eos: int = Form(10),
    save_as: str | None = Form(None),
    voice_label: str | None = Form(None),
    persist: bool = Form(True),
    saved_voice: str | None = Form(None),
):
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional built-in voice name (e.g., "alba"), or voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with the others)
        frames_after_eos: Extra 80ms frames generated after end-of-speech is detected.
            Higher values leave more trailing audio so the last word/breath isn't cut off.
        save_as: Optional filename (without extension) to save the result under, in
            STORAGE_DIR. If omitted, an auto-incrementing "adc_NNN" name is used.
        voice_label: Optional human-readable label of which voice was used (metadata only).
        persist: Whether to save the result to disk at all. Set to false for
            throwaway/intermediate generations (e.g. individual dialogue lines that
            will be merged client-side into one final file).
        saved_voice: Filename of a voice previously saved via POST /voices (e.g.
            "yo_alegre.wav"). Lets other local apps clone by name without having to
            upload the reference audio on every call.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    provided = [v for v in (voice_url, voice_wav, saved_voice) if v is not None]
    if len(provided) > 1:
        raise HTTPException(
            status_code=400, detail="Provide only one of voice_url, voice_wav, or saved_voice"
        )
    if not provided:
        voice_url = get_default_voice_for_language(str(tts_model.origin))

    # Use the appropriate model state
    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in _ORIGINS_OF_PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400, detail="voice_url must start with http://, https://, or hf://"
            )
        model_state = tts_model._cached_get_state_for_audio_prompt(voice_url)
        logging.warning("Using voice from URL: %s", voice_url)
    elif saved_voice is not None:
        voice_path = _safe_voice_path(saved_voice)
        if not voice_path.is_file():
            raise HTTPException(status_code=404, detail=f"Saved voice not found: {saved_voice}")
        model_state = tts_model.get_state_for_audio_prompt(voice_path, truncate=True)
    elif voice_wav is not None:
        # Use uploaded voice file - preserve extension for format detection
        suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name

        # Close the file before reading it back (required on Windows)
        try:
            model_state = tts_model.get_state_for_audio_prompt(Path(temp_file_path), truncate=True)
        finally:
            os.unlink(temp_file_path)
    else:
        raise HTTPException(status_code=500, detail="This should never happen.")

    save_path = None
    if persist:
        save_path = _resolve_save_path(save_as)
        _write_metadata(save_path, text=text, label=voice_label, kind="single")

    headers = {"Transfer-Encoding": "chunked"}
    if save_path is not None:
        headers["Content-Disposition"] = f"attachment; filename={save_path.name}"
        headers["X-Saved-Filename"] = save_path.name

    return StreamingResponse(
        generate_data_with_state(text, model_state, frames_after_eos, save_path),
        media_type="audio/wav",
        headers=headers,
    )


@web_app.get("/audios")
def list_audios():
    """Lista los audios guardados en disco -- para que otras apps locales los consulten."""
    files = sorted(STORAGE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified_at": f.stat().st_mtime,
            **_read_metadata(f),
        }
        for f in files
    ]


@web_app.get("/audios/{filename}")
def get_audio(filename: str):
    path = _safe_storage_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio no encontrado")
    return FileResponse(path, media_type="audio/wav")


@web_app.delete("/audios/{filename}")
def delete_audio(filename: str):
    path = _safe_storage_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio no encontrado")
    path.unlink()
    meta_path = path.with_suffix(".json")
    if meta_path.is_file():
        meta_path.unlink()
    return {"status": "deleted"}


@web_app.post("/audios")
def upload_audio(
    audio: UploadFile = File(...),
    save_as: str | None = Form(None),
    text: str | None = Form(None),
    voice_label: str | None = Form(None),
    kind: str = Form("single"),
):
    """Guarda un .wav ya generado en el cliente (ej: el diálogo multi-voz ya unido)."""
    save_path = _resolve_save_path(save_as)
    with open(save_path, "wb") as f:
        f.write(audio.file.read())
    _write_metadata(save_path, text=text, label=voice_label, kind=kind)
    return {"filename": save_path.name}


@web_app.get("/voices")
def list_voices():
    """Lista las voces guardadas -- para clonar por nombre sin volver a subir el audio."""
    files = sorted(VOICES_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    return [
        {
            "name": f.stem,
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified_at": f.stat().st_mtime,
        }
        for f in files
    ]


@web_app.get("/voices/{filename}")
def get_voice(filename: str):
    path = _safe_voice_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Voz no encontrada")
    return FileResponse(path, media_type="audio/wav")


@web_app.post("/voices")
def upload_voice(
    name: str = Form(...),
    voice_wav: UploadFile = File(...),
):
    """Guarda una muestra de voz con nombre, para reutilizar luego sin volver a subirla."""
    save_path = _resolve_voice_path(name)
    with open(save_path, "wb") as f:
        f.write(voice_wav.file.read())
    return {"name": save_path.stem, "filename": save_path.name}


@web_app.delete("/voices/{filename}")
def delete_voice(filename: str):
    path = _safe_voice_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Voz no encontrada")
    path.unlink()
    return {"status": "deleted"}


@cli_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language for the TTS model. "
            "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l', 'portuguese', 'italian', 'spanish'."
            " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Start the FastAPI server."""

    global tts_model
    tts_model = TTSModel.load_model(language=language, config=config, quantize=quantize)

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[str, typer.Option(help="Text to generate")] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to audio conditioning file (voice to clone). "
                "Defaults to a built-in voice chosen from the language: "
                "'giovanni' for italian, 'lola' for spanish, 'juergen' for german, "
                "'rafael' for portuguese, 'estelle' for french, 'alba' otherwise."
            ),
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'spanish_24l',"
                "'german_24l', 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview. They're not the final "
                "models for those languages."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float | None,
        typer.Option(
            help="Temperature for generation. Defaults to the model's recommended "
            "value from its config (0.3 for the English model, 0.7 otherwise)."
        ),
    ] = None,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens per chunk.")
    ] = MAX_TOKEN_PER_CHUNK,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Generate speech using Kyutai Pocket TTS."""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text is None:
            text = get_default_text_for_language(language)
        if text == "-":
            # Read text from stdin
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temperature,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
        )
        tts_model.to(device)

        if voice is None:
            voice = get_default_voice_for_language(language)
        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


# ----------------------------------------------
# export audio to safetensors CLI implementation
# ----------------------------------------------


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l','spanish_24l',"
                " 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
):
    """Convert and save audio to .safetensors file"""

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True
        )
        export_model_state(model_state, export_path)


if __name__ == "__main__":
    cli_app()
