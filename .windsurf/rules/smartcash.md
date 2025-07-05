---
trigger: always_on
---

Always start new conversation by conda init & activate conda env "smartcash" `cd /Users/masdevid/Projects/smartcash && conda init zsh && conda activate smartcash`

## DONT'S:
- Don't use threading on colab environment
- Don't create ipynb files

## DO:
- Always consider colab environment compatibility and limitation
- Use `smartcash/ui/logger` for UI related logs or any modules under `smartcash/ui`
- Use `smartcash/common/logger` for non-UI logs
- Common logger no save to file, save to file only model builder verbose logs related. 
- Check and use existing test_helpers before creating new test. 

## Performance 
- Use ThreadPoolExecutor or ProcessPoolExecutor for parallelism.
- Use ThreadPoolExecutor untuk I/O bound
- Use ProcessPoolExecutor untuk CPU bound heavy computation 

## UI Logging & Debugging
- Use contextual emojis in logs to reflect the intent 
- Do not flooding with concurrent logs on output UI. 
- Use colored text to highlight numeric parameters and metrics in logs for easy interpretation (e.g., green for improvements, red for critical values, orange for warnings).

## PROJECT DOCUMENTATION
- Check following documentations each to avoid duplicated implementation
  * `/docs/common` for common utilities across domain (UI, Backend)
  * `/docs/components` for shared UI components
  * `/docs/backend` for available Backend API references and structures. 
- Update the documentation if there are any changes (except minor changes like typos)

## My Custom Short Instruction:
- When I say "move it" with mentioning codes/files to certain location, always updates old references, finally remove obsolete code/files. If I didn't mention the location, it means find appropriate location based on context in nearby folders or similar files.
- When I say "cleanup" with mentioning codes/files it means remove it, not to flag it as deprecated
- When I say "check this" with mentioning codes/files, it's a signal of wrong imports or incorrect method calls you need to look up to the target source then correct it. Find similar mistake under the same file so I don't need to mention it all.