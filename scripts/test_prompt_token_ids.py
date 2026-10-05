"""Local CPU fixtures only; no tokenizer/model import or remote execution."""
import json
from prompt_token_ids import normalize_prompt_ids
class TensorFixture:
    def __init__(self,x):self.x=x
    def tolist(self):return self.x
class BatchEncodingFixture(dict):pass
valid=[[11,22,33],[[11,22,33]],TensorFixture([11,22,33]),TensorFixture([[11,22,33]]),{'input_ids':[11,22,33],'attention_mask':[1,1,1]},BatchEncodingFixture(input_ids=TensorFixture([[11,22,33]]),attention_mask=TensorFixture([[1,1,1]]))]
for x in valid:
    ids=normalize_prompt_ids(x);assert ids==[11,22,33] and len(ids)==3
    assert normalize_prompt_ids(json.loads(json.dumps(ids)))==ids
invalid=[[],[[]],[[1],[2]],[[[1]]],[True],[1.0],['1'],[-1],{'attention_mask':[1]}, {'input_ids':[]}, {'input_ids':[[1],[2]]},(1,2),None,[1]*513,TensorFixture([[1],[2]])]
for x in invalid:
    try:normalize_prompt_ids(x)
    except ValueError:pass
    else:raise AssertionError(repr(x))
assert len(normalize_prompt_ids([1]*512))==512
print(json.dumps({'valid_return_fixtures':len(valid),'malformed_rejected':len(invalid),'boundary_512_valid':True,'serialization_roundtrips':len(valid),'tensor_fixture':'tolist tensor protocol; no torch dependency or model import','model_calls':0,'remote_attempts':0}))
