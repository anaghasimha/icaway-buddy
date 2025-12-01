# file: notion_page_blocks.py
import os
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = os.getenv("NOTION_PAGE_ID")

if not NOTION_TOKEN or not PAGE_ID:
    raise SystemExit("Set NOTION_TOKEN and NOTION_PAGE_ID in .env")

notion = Client(auth=NOTION_TOKEN)

def list_block_types(page_id: str):
    res = notion.blocks.children.list(block_id=page_id)
    return [b["type"] for b in res["results"]]

def append_paragraph(page_id: str, text: str):
    notion.blocks.children.append(
        block_id=page_id,
        children=[{
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }]
    )

if __name__ == "__main__":
    print("Existing block types:", list_block_types(PAGE_ID))
    append_paragraph(PAGE_ID, "Hello from Python 👋")
    print("Appended a paragraph.")
