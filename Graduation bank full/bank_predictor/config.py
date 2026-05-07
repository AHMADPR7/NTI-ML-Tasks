from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "bank-full.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "bank_marketing_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "bank_marketing_metadata.json"

RANDOM_STATE = 42
TARGET_COLUMN = "y"
LEAKY_COLUMNS = ("duration",)
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"
POSITIVE_SAMPLE_WEIGHT = 1.5

FEATURE_FIELDS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
)

NUMERIC_FIELDS = (
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
)

JOB_OPTIONS = (
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
)

MARITAL_OPTIONS = ("divorced", "married", "single")
EDUCATION_OPTIONS = ("primary", "secondary", "tertiary", "unknown")
YES_NO_OPTIONS = ("no", "yes")
CONTACT_OPTIONS = ("cellular", "telephone", "unknown")
MONTH_OPTIONS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
POUTCOME_OPTIONS = ("failure", "other", "success", "unknown")

CATEGORICAL_OPTIONS = {
    "job": JOB_OPTIONS,
    "marital": MARITAL_OPTIONS,
    "education": EDUCATION_OPTIONS,
    "default": YES_NO_OPTIONS,
    "housing": YES_NO_OPTIONS,
    "loan": YES_NO_OPTIONS,
    "contact": CONTACT_OPTIONS,
    "month": MONTH_OPTIONS,
    "poutcome": POUTCOME_OPTIONS,
}

CATEGORICAL_FIELDS = tuple(CATEGORICAL_OPTIONS.keys())

FIELD_LABELS = {
    "age": "Age",
    "job": "Job",
    "marital": "Marital Status",
    "education": "Education",
    "default": "Has Credit in Default",
    "balance": "Average Yearly Balance",
    "housing": "Has Housing Loan",
    "loan": "Has Personal Loan",
    "contact": "Contact Communication Type",
    "day": "Last Contact Day of Month",
    "month": "Last Contact Month",
    "campaign": "Contacts During Campaign",
    "pdays": "Days Since Previous Contact",
    "previous": "Contacts Before This Campaign",
    "poutcome": "Previous Campaign Outcome",
}

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL_ENV = "BANK_API_URL"
