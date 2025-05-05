# Simplification Examples

This document provides examples of text simplification using CasaLingua's simplification API.

## Basic Simplification

### Request

```python
import requests
import json

url = "http://localhost:8000/pipeline/simplify"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "text": "The utilization of simplified language enhances comprehension for individuals across diverse educational backgrounds.",
    "language": "en",
    "target_level": "simple"
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Original: {data['text']}")
print(f"Simplified: {result['data']['simplified_text']}")
```

### Response

```json
{
  "status": "success",
  "data": {
    "original_text": "The utilization of simplified language enhances comprehension for individuals across diverse educational backgrounds.",
    "simplified_text": "Using simple language helps people with different education levels understand better.",
    "language": "en",
    "target_level": "simple",
    "process_time": 0.234
  }
}
```

## Legal Housing Document Simplification

### Request

```python
import requests
import json

url = "http://localhost:8000/pipeline/simplify"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant, its agents, servants, employees, licensees, visitors, invitees, and any of the tenant's contractors and subcontractors.",
    "language": "en",
    "target_level": "simple",
    "domain": "legal-housing",
    "verify_output": True,
    "auto_fix": True
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Original: {data['text']}")
print(f"Simplified: {result['data']['simplified_text']}")
if "verification" in result["data"]:
    print(f"Verification Score: {result['data']['verification']['score']}")
```

### Response

```json
{
  "status": "success",
  "data": {
    "original_text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant, its agents, servants, employees, licensees, visitors, invitees, and any of the tenant's contractors and subcontractors.",
    "simplified_text": "The Tenant must protect the Landlord from all claims, lawsuits, and demands caused by the Tenant's careless or deliberate actions. This includes actions by the Tenant's employees, guests, visitors, and contractors.",
    "language": "en",
    "target_level": "simple",
    "verification": {
      "verified": true,
      "score": 0.82,
      "confidence": 0.78,
      "metrics": {
        "word_count_original": 55,
        "word_count_simplified": 31,
        "word_ratio": 0.56,
        "semantic_similarity": 0.88
      }
    }
  }
}
```

## Simplification with Specific Grade Level

### Request

```python
import requests
import json

url = "http://localhost:8000/pipeline/simplify"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "text": "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate.",
    "language": "en",
    "target_level": "4"  # 4th grade reading level
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Original: {data['text']}")
print(f"Simplified: {result['data']['simplified_text']}")
```

### Response

```json
{
  "status": "success",
  "data": {
    "original_text": "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate.",
    "simplified_text": "Mitochondria are tiny parts of cells that make energy. The cell uses this energy to do its work. The energy is stored in a small molecule called ATP.",
    "language": "en",
    "target_level": "4",
    "process_time": 0.312
  }
}
```

## Simplifying Longer Documents

For longer documents, you can simplify them in chunks and then combine the results:

```python
import requests
import json

def simplify_text_in_chunks(text, chunk_size=1000, overlap=100):
    """Simplify a long text by breaking it into chunks."""
    
    # Break text into chunks
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 10:  # Ensure chunk is substantial
            chunks.append(chunk)
    
    # Simplify each chunk
    simplified_chunks = []
    for chunk in chunks:
        response = requests.post(
            "http://localhost:8000/pipeline/simplify",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_API_KEY"
            },
            data=json.dumps({
                "text": chunk,
                "language": "en",
                "target_level": "simple",
                "domain": "legal-housing"
            })
        )
        result = response.json()
        simplified_chunks.append(result["data"]["simplified_text"])
    
    # Combine simplified chunks
    return " ".join(simplified_chunks)

# Example usage
long_document = """[Long legal document text here...]"""
simplified_document = simplify_text_in_chunks(long_document)
print(simplified_document)
```

## Batch Simplification

For multiple texts at once:

```python
import requests
import json

url = "http://localhost:8000/pipeline/batch"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "operation": "simplify",
    "items": [
        {
            "text": "The lessor reserves the right to access the premises for inspection purposes with 24 hours advance notice.",
            "language": "en",
            "target_level": "simple",
            "domain": "legal-housing"
        },
        {
            "text": "Tenant shall be responsible for the payment of all utilities and services to the premises.",
            "language": "en",
            "target_level": "simple",
            "domain": "legal-housing" 
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
results = response.json()

for i, result in enumerate(results["data"]["results"]):
    print(f"Item {i+1}:")
    print(f"Original: {data['items'][i]['text']}")
    print(f"Simplified: {result['simplified_text']}")
    print()
```

## Error Handling

### Handling Text Too Long

```python
import requests
import json

url = "http://localhost:8000/pipeline/simplify"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "text": "[Extremely long text that exceeds maximum length]",
    "language": "en",
    "target_level": "simple"
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    result = response.json()
    print(f"Simplified: {result['data']['simplified_text']}")
except requests.exceptions.HTTPError as e:
    error_data = e.response.json()
    if error_data["error"]["code"] == "TEXT_TOO_LONG":
        print("Error: Text is too long. Please break it into smaller chunks.")
    else:
        print(f"Error: {error_data['message']}")
```

## Command Line Examples

### Using cURL

```bash
curl -X POST http://localhost:8000/pipeline/simplify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims.",
    "language": "en",
    "target_level": "simple",
    "domain": "legal-housing"
  }'
```

### Using jq to Parse Results

```bash
curl -X POST http://localhost:8000/pipeline/simplify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims.",
    "language": "en",
    "target_level": "simple",
    "domain": "legal-housing"
  }' | jq '.data.simplified_text'
```

## JavaScript Example

```javascript
async function simplifyText() {
  const url = "http://localhost:8000/pipeline/simplify";
  const data = {
    text: "The tenant shall indemnify and hold harmless the landlord from and against any and all claims.",
    language: "en",
    target_level: "simple",
    domain: "legal-housing"
  };
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
      },
      body: JSON.stringify(data)
    });
    
    const result = await response.json();
    
    if (result.status === "success") {
      console.log("Original:", data.text);
      console.log("Simplified:", result.data.simplified_text);
    } else {
      console.error("Error:", result.message);
    }
  } catch (error) {
    console.error("Request failed:", error);
  }
}

simplifyText();
```