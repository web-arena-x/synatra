# Data Generation

## Data generation from Wikihow
[wikihow](./wikihow)

1. Filter out articles that are not about browser tasks (`gpt-3.5`)
2. Generation step 1 (`generate_data.py`): use `gpt-4` to make the article concrete. The prompt is in `prompts/prompt_step1.yaml`, the version is latest
3. Generation step 2 (`generate_data.py`): Sample a break point and generate an HTML which is the result of previous steps, and is the input of the next step. The interactive component is marked with `id="next-action-target-element"`. The prompt is in `prompts/prompt_step2.yaml`, the version is latest.

### Result
`metis: /projects/metis2/users/shuyanzh/agent_model_data/wikihow/wikihow_2000.jsonl`
```python
{
    'task': <the intent>
    'prev_actions': <all previous actions, in the form of Python code>
    'next_action': <the next action, in the form of Python code>
    'html': <the generated HTML, the target element is marked with id="next-action-target-element">
}
```

### Misc
`data_inspect.py` can turn the generated data to HTML for easier visuailization.

## Data generation from HTML
[clueweb](./clueweb)

1. Extract HTML from clueweb, get the corresponding AXTree. Sample a subtree with `N~(125, 50)`
2. Generation (`generate_data.py`): use `gpt-4` to generate the task, the previous actions and the next action. The interactive component is indicated as `element_id=` in the next action.

### Result
`metis: /projects/metis2/users/shuyanzh/agent_model_data/clueweb/en0000-01.2000.jsonl`
```python
{
    'task': <the intent>
    'prev_actions': <all previous actions, in the form of Python code>
    'next_action': <the next action, in the form of Python code>
    'ax_tree': the AXTree of the HTML
}
```
