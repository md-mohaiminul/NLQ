from dataloaders.data_loader_ego4d_negative_retrieval import Ego4d_DataLoader
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
import timeit
from torch.utils.data import Dataset, DataLoader

tokenizer = ClipTokenizer()

dataset = Ego4d_DataLoader(subset = 'train', data_path = '/playpen-storage/mmiemon/ego4d/data/annotations',
                           features_path = '/playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224',
                           tokenizer = tokenizer, max_frames=12)

print(len(dataset))
if __name__ == '__main__':
    # pairs_text, pairs_mask, pairs_segment, video, video_mask, negative_videos, negative_video_masks = dataset.__getitem__(100)
    dataloader = DataLoader(
        dataset,
        batch_size=6,
        num_workers=12
    )
    start = timeit.default_timer()
    pairs_text, pairs_mask, pairs_segment, video, video_mask, negative_videos, negative_video_masks = next(iter(dataloader))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    # for batch in dataloader:
    #     pairs_text, pairs_mask, pairs_segment, video, video_mask, negative_videos, negative_video_masks = batch
    #     print(pairs_text.shape, pairs_mask.shape, pairs_segment.shape, video.shape, video_mask.shape, negative_videos.shape, negative_video_masks.shape)
    #     break