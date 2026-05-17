import sys
import numpy as np
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.numeric'] = numpy.core.numeric

sys.path.insert(0, 'src')
from collaborative_filtering import UserBasedCF
try:
    ubcf = UserBasedCF.load('models/saved/ubcf.pkl')
    print("Success loading ubcf.pkl")
    print(ubcf)
except Exception as e:
    import traceback
    traceback.print_exc()
