# VANTAGE Benchmark Documentation

## VANTAGE Temporal
The VANTAGE Temporal benchmark evaluates a model’s ability to pinpoint the exact duration of specific, domain-relevant events within a continuous video stream. Unlike standard action recognition, this task requires precise temporal boundaries (start/end times) for events that may be subtle or complex, such as traffic violations or warehouse safety incidents.

### Prompt Format

**Template:**

    {Question} Please provide the result in json format with 'mm:ss.ss' format for time depiction for the event. Use keywords 'start', 'end' in the json output.

**Example:**
    
    At what point does a dark red car entering from the right of the frame drive ahead and turn right without a proper right turn signal happen in the video? Please provide the result in json format with 'mm:ss.ss' format for time depiction for the event. Use keywords 'start', 'end' in the json output.

| Feature | Requirement |
| :--- | :--- |
| **Time Format** | `mm:ss.ss` |
| **Output Type** | Strict JSON |
| **Required Keys** | `"start"`, `"end"` |

#### Example JSON Output
```json
{
  "start": "00:08.93",
  "end": "00:19.06"
}
```
---

## VANTAGE Event Verification
This benchmark serves as a diagnostic tool for semantic understanding. It presents the model with a video and a factual statement regarding an incident, requiring a binary Yes/No confirmation. It evaluates whether the model truly "perceives" the outcome of a physical interaction rather than just identifying the objects involved.

### Prompt Format
**Template:**

    You are an event verification assistant. Given a video and a statement about an event, determine whether the statement is true or false. Answer with 'Yes' if the statement is true, or 'No' if it is false.

    Question: {Question}

**Example:**
    
    You are an event verification assistant. Given a video and a statement about an event, determine whether the statement is true or false. Answer with 'Yes' if the statement is true, or 'No' if it is false.

    Question: Does the forklift falls over and crashes to the ground?

| Feature | Requirement |
| :--- | :--- |
| **Expected Output** | "Yes" or "No" |
---

## VANTAGE VQA
VANTAGE VQA is the suite's most comprehensive semantic benchmark. It uses a Multiple-Choice Question (MCQ) format to test high-level reasoning across four dimensions: spatial relationships, temporal order, object counting, and causal reasoning.

### Prompt Format
**Template:**
```text
You are provided with a sequence of video frames depicting a scene. 
Begin with a concise overview of what's happening; keep items conceptual, not implementation-level. 
Answer the question based only on the visual content of the image.
Question: {Question}
Select your answer from the choices below:
{Options}
Respond with ONLY the letter corresponding to your answer (A, B, C, or D). Do not provide any explanation or other text.
```

**Example:**
```text
You are provided with a sequence of video frames depicting a scene. 
Begin with a concise overview of what's happening; keep items conceptual, not implementation-level. 
Answer the question based only on the visual content of the image.
Question: How many people are walking by the forklift?
Select your answer from the choices below:
A. 1
B. 2
C. 3
D. 4
Respond with ONLY the letter corresponding to your answer (A, B, C, or D). Do not provide any explanation or other text.
```

| Feature | Requirement |
| :--- | :--- |
| **Expected Output** | "A", "B", "C", or "D" |
---

## VANTAGE Spatial
**Astro2D**: Focuses on fine-grained human bounding box prediction. It requires the model to precisely localize people within high-definition industrial or urban frames, providing a baseline for downstream safety tasks.

**VANTAGE 2D Detection**: A standard class-based localization task. It requires the model to detect and categorize multiple object classes within the automobile category simultaneously across diverse scene layouts.

**2D Grounding**: A complex language-vision alignment task. The model must identify a specific target object(s) based on a descriptive natural language prompt.

**2D Spatial Pointing**: It evaluates fine-grained spatial accuracy by requiring the model to choose between multiple candidate coordinates in a multiple-choice format, testing whether the model can point to the correct visual referent.


### Astro2D Prompt
```text
Locate every instance that belongs to the following categories: 'person'. Report bbox coordinates in JSON format.
```

#### Example JSON Output for Astro2D
```text
[
	{"bbox_2d": [904, 638, 925, 728], "label": "person"},
	{"bbox_2d": [919, 638, 937, 702], "label": "person"}
]
```
---

### VANTAGE 2D Detection
```text
Locate every instance that belongs to the following categories: 'sedan, SUV, bus, truck'. For each instance of the class, report bbox coordinates in JSON format. Do not group instances and report only individual instances.
```

### Example JSON Output for VANTAGE 2D Detection
```text
[
	{"bbox_2d": [304, 367, 339, 388], "label": "car"},
	{"bbox_2d": [344, 369, 376, 388], "label": "truck"}
]
```
---



### 2D Grounding
**Template:**
```text
As an AI visual assistant, your task is to identify and locate specific objects in the provided image.

Supplied Description: {description}

Task:
Based on the description and the image content, identify the key groups of objects mentioned. For each group, provide a descriptive label and the precise bounding box coordinates for every individual instance in that group. 

Coordinates must be normalized to a 0-1000 scale in [x1, y1, x2, y2] format.

Output Format:
Return your findings as a JSON-like list of strings, following this exact format:
The [object description]: [[x1, y1, x2, y2], [x3, y3, x4, y4]]

Example:
The blue cars parked on the right: [[579, 454, 690, 636], [342, 441, 435, 608]]
```
**Example:**
```text
As an AI visual assistant, your task is to identify and locate specific objects in the provided image.

Supplied Description: The black cars are stopped at the red traffic light on the left side of the road.

Task:
Based on the description and the image content, identify the key groups of objects mentioned. For each group, provide a descriptive label and the precise bounding box coordinates for every individual instance in that group. 

Coordinates must be normalized to a 0-1000 scale in [x1, y1, x2, y2] format.

Output Format:
Return your findings as a JSON-like list of strings, following this exact format:
The [object description]: [[x1, y1, x2, y2], [x3, y3, x4, y4]]

Example:
The blue cars parked on the right: [[579, 454, 690, 636], [342, 441, 435, 608]]
```

### Example Outputs for 2D Grounding 
List of bounding boxes
```text
[[100, 200, 300, 400], [50, 60, 200, 300]]
```

Dictionary with bbox key
```text
{"bbox_2d": [[100, 200, 300, 400], [50, 60, 200, 300]]}
```

List of dictionaries
```text
[
  {"bbox": [100, 200, 300, 400]},
  {"bbox": [50, 60, 200, 300]}
]
```

---

### 2D Spatial Pointing

**Template:**
```text
Answer the following spatial grounding question based on the image.
Question: {Question}
Options (Coordinates are [x,y]):
{Opitions}
Respond with ONLY the letter of the correct option (A, B, C, or D).
```

**Example:**
```text
Answer the following spatial grounding question based on the image.
Question: Which car is the leftmost one in the scene?
Options (Coordinates are [x,y]):
A. 137,778
B. 596,783
C. 40,771
D. 629,777
Respond with ONLY the letter of the correct option (A, B, C, or D).
```
---


The safest bet to ensure model outputs are parseable is to follow the table below.

| Feature | Requirement |
| :--- | :--- |
| **Bounding Box Format** | `[x1, y1, x2, y2]` |
| **Output Type** | List of JSON Objects |
| **Required Keys** | `"bbox_2d"`, `"label"` |


---



