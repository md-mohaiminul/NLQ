from torch.utils.data import Dataset
from clip4clip_reranking2 import _get_rawvideo

class PredictionDataset(Dataset):
    def __init__(self,pred_datum, topK):
        self.pred_datum = pred_datum
        self.topK = topK

    def __len__(self):
        return topK

    def __getitem__(self, idx):
        s = math.floor(pred_datum['predicted_times'][i][0])
        e = min(math.ceil(pred_datum['predicted_times'][i][1]), s + 30)
        video, video_mask = _get_rawvideo(args, rawVideoExtractor, choice_video_ids, s, e)

        return video, video_mask