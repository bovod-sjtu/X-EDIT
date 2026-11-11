# Seed-VC service integration

The controller communicates over HTTP. Inference is deployed from the pinned
edit-data-pipeline submodule; official SDK source remains unmodified.
See [deployment](../../../../docs/DEPLOYMENT.md) and
[source boundaries](../../../../docs/LICENSES.md).

```bash
python deployment/pipeline.py fetch seed-vc
python deployment/pipeline.py check seed-vc
```

Use `deployment/bootstrap.py` to install and launch the corresponding service.
A source check is not an inference or quality check.
