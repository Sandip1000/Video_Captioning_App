import numpy as np
import av
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", use_fast = True)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



def sample_frame_indices(container, clip_len=8):
     stream = container.streams.video[0]
     total_frames = stream.frames
     indices = np.linspace(0, max(total_frames - 1, 0), num=clip_len).astype(np.int64)
     return indices




def get_video_tensor(video_path):
    container = av.open(video_path)

    # Sample frame indices
    indices = sample_frame_indices(container, clip_len=8)

    # Read video frames (T,H,W,3)
    video_numpy = read_video_pyav(container, indices)

    # Process frames with image processor directly
    video_tensor = processor(list(video_numpy), return_tensors="pt")['pixel_values'].squeeze(0)  # (T, 3, H, W)

    return video_tensor