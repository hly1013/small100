"""inference"""


from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer
import os

# for path
cur_dir_path = os.path.dirname(__file__)
if cur_dir_path == '': cur_dir_path = '.'

# * user should set this * #
FINE_TUNED = True
if FINE_TUNED:
    model_name = '200'
else:
    model_name = None


def main(**kwargs):
    # set model
    if FINE_TUNED:
        model = M2M100ForConditionalGeneration.from_pretrained(f'{cur_dir_path}/model/{model_name}')
    else:
        model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")

    # set tokenizer
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

    # function for ENKO translation test
    def translate_enko():
        english_text = input('english text?\n').rstrip()
        tokenizer.tgt_lang = 'ko'
        encoded_en = tokenizer(english_text, return_tensors='pt')
        generated_tokens = model.generate(**encoded_en)
        korean_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return korean_output

    # call the test function -> should input English text
    result = translate_enko()[0]
    print(result)


if __name__ == "__main__":
    settings = {'FINE_TUNED': FINE_TUNED, 'model_name': model_name}
    main(**settings)
