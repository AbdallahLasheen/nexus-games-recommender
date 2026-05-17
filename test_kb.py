import sys
import numpy as np
import pandas as pd
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.numeric'] = numpy.core.numeric

sys.path.insert(0, 'src')
from knowledge_based import KnowledgeBasedRecommender

items_df = pd.read_parquet('data/processed/items_full.parquet', columns=["parent_asin", "title", "price", "categories"])
print("Columns in items_df:", items_df.columns)

kb = KnowledgeBasedRecommender.load('models/saved/kb.pkl', items_df=items_df)
print("Columns in kb.items_df after load:", kb.items_df.columns)
