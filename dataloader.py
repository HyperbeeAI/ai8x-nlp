###########################################################################
# NLP demo software by HyperbeeAI.                                        #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
license_statement = "NLP demo software by HyperbeeAI. Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai"
print("imported dataloader.py")
print(license_statement)
print("")

from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy.data import Field, BucketIterator
import os

class NewsDataset(TranslationDataset):

    name = 'news-comm-v15'

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    @classmethod
    def splits(cls, exts, fields, root='./',
               train='news-comm-v15-all', validation='news-comm-v15-all-valid', test='news-comm-v15-all-test', **kwargs):

        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(NewsDataset, cls).splits(exts, fields, path, root, train, validation, test, **kwargs)
