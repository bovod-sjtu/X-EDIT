# QuarkAudio-UniSE Backend

Backend service for `tse.extract_target_speaker`.

UniSE supports target speaker extraction using reference enrollment audio. The X-Editor `tse` wrapper requires `reference_audio_path`; natural-language-only target hints are not enough for this backend.

The source release contains the original HTTP wrapper only. Obtain the upstream
implementation separately under its own terms:

```text
https://github.com/alibaba/unified-audio/tree/main/QuarkAudio-UniSE
```

Set `QUARKAUDIO_UNISE_REPO_DIR` to the external `QuarkAudio-UniSE` checkout.
Required files relative to that checkout are:

- `checkpoints/epoch=20-step=109367.ckpt`
- `checkpoints/BiCodec/config.yaml`
- `checkpoints/BiCodec/model.safetensors`
- `checkpoints/wav2vec2-large-xlsr-53/config.json`

The BiCodec and wav2vec files are expected from `SparkAudio/Spark-TTS-0.5B` as described by the UniSE README.
No implementation, checkpoint or generated audio is bundled. This optional
integration is outside the managed 29-service stack; record and review the exact
source and weight revisions before provisioning it.

Run the service wrapper:

```bash
cd x_editor/tools/catalog/quarkaudio_unise
./setup.sh
export QUARKAUDIO_UNISE_REPO_DIR=/data/external/unified-audio/QuarkAudio-UniSE
.venv/bin/python service.py --host 127.0.0.1 --port 8025
```

`setup.sh` installs the FastAPI wrapper dependencies by default, using a uv
venv with system site packages so the service process can reuse already
installed heavy runtime packages without importing them in the main framework.
The `/health` endpoint reports `not_ready` until both checkpoints and model
runtime imports are available. To install the minimal UniSE inference
dependencies into this venv:

```bash
X_EDITOR_QUARKAUDIO_INSTALL_FULL_DEPS=1 ./setup.sh
```
