# Fix Model Loading Issues in CasaLingua Pipeline

## Summary
- Fixed critical model loading issues that prevented server startup
- Created proper model registry configuration with correct model classes
- Fixed scope issues with transformer imports to prevent UnboundLocalError
- Ensured all models use the correct classes for their intended tasks

## Test plan
- [x] Server starts up successfully with all models loaded
- [x] Health endpoints report all models as healthy
- [x] Language detection endpoints are functional
- [x] Translation endpoints are functional
- [x] Hardware detection and model configuration works for Apple Silicon