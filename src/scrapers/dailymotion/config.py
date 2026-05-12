# ── Search terms ────────────────────────────────────────────────────────────
SEARCH_QUERIES = [
    "cat meowing",
    "cat purring",
    "cat chattering",
    "kitten sounds",
    "cat vocalization",
    "cat talking",
    "funny cat sounds",
    "cat hissing",
    "cat trilling",
]

# ── Dailymotion API ─────────────────────────────────────────────────────────
DM_API_BASE = "https://api.dailymotion.com"
DM_RESULTS_PER_PAGE = 100
DM_MAX_PAGES_PER_QUERY = 5

DM_FIELDS = ",".join(
    [
        "id",
        "title",
        "duration",
        "url",
        "tags",
        "channel",
        "language",
        "views_total",
        "likes_total",
        "created_time",
        "description",
    ]
)

# ── Post-level quality gates (before any filtering stages) ─────────────────
MIN_VIEWS = 500
MIN_VIDEO_DURATION_SEC = 2
MAX_VIDEO_DURATION_SEC = 180

# ── Metadata tag filter (tag_filter.py) ────────────────────────────────────
# Title/description/tags (lowercased) containing any of these → discard
REJECT_TITLE_KEYWORDS = [
    "fortnite",
    "minecraft",
    "roblox",
    "gta ",
    "call of duty",
    "valorant",
    "among us",
    "pokemon go",
    "walkthrough",
    "gameplay",
    "speedrun",
    "reaction only",
    "compilation #shorts spam",
]
REJECT_SUBSTRING_EXTRA = [
    " fortnite",
    " minecraft",
]

# At least one token must appear in combined title+description+tags (lowercased)
CAT_SIGNAL_KEYWORDS = [
    "cat",
    "cats",
    "kitten",
    "kittens",
    "feline",
    "kitty",
    "meow",
    "purr",
    "gato",
    "gatos",
    "chat ",  # French "cat" as word
    "gatto",
]

# ── GPT filter (gpt_filter.py) ────────────────────────────────────────────
GPT_MODEL = "gpt-4o-mini"
GPT_BATCH_SIZE = 8
GPT_MAX_RETRIES = 3
GPT_TEMPERATURE = 0.0
# Concurrent API calls for metadata batches (sync OpenAI client; reduce if rate-limited).
GPT_PARALLEL_WORKERS = 5

# ── Paths (relative to cwd when running from dailymotion_scraper/) ──────────
# Video + audio live in the same folder (matches YouTube/TikTok pipelines).
VIDEO_DIR = "output/videos"
AUDIO_DIR = "output/videos"
METADATA_DIR = "output/metadata"
# Resolved via utils.load_pipeline_config (repo root or package dir).
DEFAULT_PIPELINE_CONFIG = "config/pipeline.yaml"

# ── Download behaviour ─────────────────────────────────────────────────────
SLEEP_BETWEEN_REQUESTS = 1
SLEEP_BETWEEN_DOWNLOADS = 2

# ── Run / reporting (YouTube-shaped pipeline parity) ─────────────────────
DEFAULT_RUN_NAME = "dailymotion_run"
GPT_COST_PER_1K_TOKENS = 0.00015  # rough estimate for gpt-4o-mini input+output blend
