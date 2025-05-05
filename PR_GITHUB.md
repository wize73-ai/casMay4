# Fix model loading and API endpoints in CasaLingua pipeline

## Summary
This PR fixes critical model loading issues in the CasaLingua pipeline and implements missing API endpoints to ensure full functionality. All models now load properly with their correct classes, and all API endpoints are accessible and working correctly.

## Changes
- Created proper model registry configuration with correct model classes
- Fixed scope issues with transformer imports to prevent UnboundLocalError
- Fixed model class references (T5ForConditionalGeneration for simplifier)
- Fixed health endpoints to properly report model registry status
- Fixed authentication bypass for development mode
- Implemented missing API endpoints for text simplification and anonymization
- Fixed language detection response format to match expected output
- Added direct model testing capability
- Added comprehensive API endpoint testing tools

## Model Status
All models now load properly with their correct classes:
- ✅ Translation: Working correctly
- ✅ Simplifier (T5): Fixed using T5ForConditionalGeneration model class
- ✅ NER Detection: Working correctly
- ✅ RAG Generator: Working correctly
- ✅ Anonymizer: Working correctly
- ✅ Language Detection: Fixed response format to include detected_language key

## API Endpoint Status
All API endpoints are now accessible and functioning:
- ✅ `/pipeline/translate`: Working correctly
- ✅ `/pipeline/detect`: Working correctly
- ✅ `/pipeline/simplify`: Implemented and working
- ✅ `/pipeline/anonymize`: Implemented and working
- ✅ `/pipeline/analyze`: Working correctly
- ✅ `/pipeline/summarize`: Working correctly

## Testing
- Added `test_direct_model_calls.py` for testing models directly without API
- Added `test_api_endpoints.py` for comprehensive Python-based API testing
- Added `test_api_endpoints.sh` for quick shell-based API testing
- All tests are passing with the fixes implemented

## Environment Information
- Hardware: Apple Silicon (M4 Max)
- OS: macOS 24.5.0
- Python: 3.10.13
- Models: T5-small, MBART-50, XLM-RoBERTa-base

## How to Test
1. Run `./bypass_auth.sh` to set the development environment
2. Start the server with `./scripts/startdev.sh`
3. Run `python test_direct_model_calls.py` to test models directly
4. Run `python test_api_endpoints.py` to test API endpoints
5. Or run `./test_api_endpoints.sh` for a quick API check

## Breaking Changes
None. All changes maintain backward compatibility with existing endpoints and models.

## Related Issues
- Fixes issue #142: Model loading errors with T5ForConditionalGeneration
- Fixes issue #156: Missing API endpoints for text simplification
- Fixes issue #157: Missing API endpoints for text anonymization
- Fixes issue #163: Language detection format inconsistency