import json, time
from pathlib import Path
import pymupdf
import chromadb
from chromadb.utils import embedding_functions

PDF_PATH = Path("data_adv/report.pdf")
OUT_DIR = Path("data_adv")
PAGES_DIR = OUT_DIR / "pages"
PAGES_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
doc = pymupdf.open(PDF_PATH)
assert len(doc) >= 100

chunks = []
stats = {
    "document": "NASA Systems Engineering Handbook (NASA/SP-2007-6105 Rev2)",
    "source_url": "https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf",
    "license": "Public Domain (U.S. Government Work)",
    "total_pages": len(doc),
    "indexed_pages": 0,
    "table_pages": [],
    "diagram_chart_pages": [],
    "skipped_pages": [],
}

for idx, page in enumerate(doc):
    pno = idx + 1
    raw_text = page.get_text("text").strip()
    imgs = [im for im in page.get_images(full=True) if im[2] >= 120 and im[3] >= 120]
    drawings = page.get_drawings()

    if len(raw_text) < 40 and not imgs:
        stats["skipped_pages"].append({"page": pno, "reason": "Blank/cover page (<40 chars, no images)"})
        continue

    stats["indexed_pages"] += 1

    # 1. Extract tables with bounding boxes
    tabs = page.find_tables()
    has_table = False
    for ti, tab in enumerate(tabs.tables):
        md = tab.to_markdown().strip()
        if md and len(md) > 25:
            has_table = True
            chunks.append({
                "id": f"nasa_p{pno:03d}_table_{ti}",
                "source_id": "nasa-se-rev2",
                "source": "data_adv/report.pdf",
                "page": pno,
                "modality": "table",
                "bbox": [round(v, 1) for v in tab.bbox],
                "has_visual": True,
                "image_path": f"data_adv/pages/page_{pno:03d}.png",
                "text": f"[TABLE Page {pno}]\n{md[:2200]}",
            })
    if has_table:
        stats["table_pages"].append(pno)

    # 2. Detect diagrams / charts (raster images or dense vector drawings)
    has_visual_fig = bool(imgs) or len(drawings) > 50
    if has_visual_fig:
        stats["diagram_chart_pages"].append(pno)

    png_rel = ""
    if has_table or has_visual_fig:
        png_file = PAGES_DIR / f"page_{pno:03d}.png"
        if not png_file.exists():
            page.get_pixmap(dpi=150).save(png_file)
        png_rel = f"data_adv/pages/page_{pno:03d}.png"

    # 3. Chunk page text (preserving figure/table captions)
    paras = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) >= 40]
    if not paras and raw_text:
        paras = [raw_text]
    for pi, para in enumerate(paras):
        chunks.append({
            "id": f"nasa_p{pno:03d}_text_{pi}",
            "source_id": "nasa-se-rev2",
            "source": "data_adv/report.pdf",
            "page": pno,
            "modality": "visual" if has_visual_fig else "text",
            "bbox": [0.0, 0.0, round(page.rect.width, 1), round(page.rect.height, 1)],
            "has_visual": bool(png_rel),
            "image_path": png_rel,
            "text": para[:1800],
        })

# 4. Build ChromaDB collection using Day 6 local all-MiniLM-L6-v2
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
)
client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("nasa_se_adv")
except Exception:
    pass
col = client.create_collection("nasa_se_adv", embedding_function=ef, metadata={"hnsw:space": "cosine"})
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    col.add(
        ids=[c["id"] for c in batch],
        documents=[c["text"] for c in batch],
        metadatas=[{"page": c["page"], "modality": c["modality"], "has_visual": c["has_visual"], "image_path": c["image_path"]} for c in batch],
    )

stats["total_chunks"] = len(chunks)
stats["index_seconds"] = round(time.perf_counter() - t0, 2)
(OUT_DIR / "chunks_adv.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT_DIR / "index_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in stats.items()}, indent=2))
