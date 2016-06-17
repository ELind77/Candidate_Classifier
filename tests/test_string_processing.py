import nose.tools as nosey
from candidate_classifier.string_processing import *


DOC1 = u"I'm starting to know what God felt like when he sat out " \
       "there in the darkness, creating the world."

DOC2 = u"And what did he feel like, Lloyd, my dear?"

DOC3 = u"Very pleased he'd taken his Valium."

DOCS = [DOC1, DOC2, DOC3]


def test_string_processor_returns_string_for_string():
    processor = StringTransformer()
    nosey.assert_equal(DOC1, processor(DOC1))


def test_string_processor_returns_list_for_string():
    processor = StringTransformer(tokenizer=string.split)
    nosey.assert_equal(DOC1.split(), processor(DOC1))


def test_string_processor_works_as_a_generator():
    processor = StringTransformer()
    nosey.assert_equal(DOCS, [d for d in processor(DOCS)])
