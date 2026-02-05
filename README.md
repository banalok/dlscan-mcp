# DLSCAN-MCP

An **MCP (Model Context Protocol) server** that exposes the full **DL-SCAN fluorescence microscopy analysis pipeline** as structured, callable tools, designed to be run interactively from **Claude Desktop** (or any MCP-capable client).

This setup allows you to run a complete DL-SCAN pipeline — preprocessing, segmentation, and quantitative analysis — **by issuing a single high-level prompt in Claude**, without writing pipeline scripts.

---

## What this repository provides

- A ready-to-run **MCP server** (`dlscan_server.py`)
- DL-SCAN-faithful preprocessing steps (Gaussian, median, brightness/contrast, CLAHE)
- Collapsed 2D image generation for segmentation (first frame or max projection)
- Optional rolling-ball background subtraction
- **StarDist 2D** segmentation with DL-SCAN-style post-processing
- Quantitative extraction of intensity and bright-pixel area over time
- Fully reproducible, file-based outputs (`.npy`, `.png`, `.csv`)

---

## Repository structure

```
DLSCAN-MCP/
├── dlscan_server.py
├── dlscan_params.yaml
├── dlscan_pipeline_prompt.txt
├── pyproject.toml
├── uv.lock
├── README.md
├── test_tiff/

```

---

## Installation (recommended: `uv`)

### 1. Install `uv`

```bash
pip install uv
```

### 2. Clone the repository

```bash
git clone https://github.com/banalok/dlscan-mcp.git
cd dlscan-mcp
```

### 3. Create environment and install dependencies

```bash
uv venv
uv sync
```

---

## Running the MCP server

```bash
uv run python dlscan_server.py
```

Leave this running while Claude Desktop connects.

---

## Using this server with Claude Desktop

Open Claude Desktop --> Settings --> Developer --> Edit Config

Create a config file:

```
claude_desktop_config.json
```

with contents:

```json
{
  "mcpServers": {
    "dlscan": {
      "command": "/path/to/uv.exe",
      "args": ["run", "python", "dlscan_server.py"],
      "cwd": "/path/to/dlscan-mcp"
    }
  }
}
```

Restart Claude Desktop after configuring.

---

## Running the DL-SCAN pipeline from Claude

Use the prompt in `dlscan_pipeline_prompt.txt`.  
Claude will read `dlscan_params.yaml`, execute preprocessing, segmentation, and extract intensity data from the input raw images.

---

## Demo video

▶ **Demo video:** 
![DL-SCAN MCP demo](https://github.com/banalok/dlscan-mcp/blob/master/demo/dlscan-mcp.mp4)

---

## Intended workflow

1. Start the MCP server
2. Connect via Claude Desktop
3. Paste the pipeline prompt
4. Review outputs
