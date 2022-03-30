import json

# nlq_train = '/playpen-storage/mmiemon/ego4d/data/annotations/nlq_val.json'
nlq_train = 'data/dataset/nlq_official_v1/val.json'
with open(nlq_train, mode="r", encoding="utf-8") as f:
    data = json.load(f)

for v in data:
    print(data[v])
    break
# for v in data['videos']:
#     #print(v)
#     print(v['video_uid'])
#     for clip in v['clips']:
#         print(clip)
#         break
#     break