import importlib
modules = ['dotenv','PyPDF2','docx','bs4','pandas','lxml','pptx','openpyxl','sentence_transformers','fuzzywuzzy','Levenshtein','xgboost','sklearn','pymongo','numpy','streamlit','nltk']
missing = []
for m in modules:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))

print('Checked modules:')
for m in modules:
    print(' -', m)

print('\nMISSING_COUNT=', len(missing))
if missing:
    print('Missing modules and errors:')
    for m, e in missing:
        print(f"{m} -> {e}")
else:
    print('All checked modules imported successfully')
