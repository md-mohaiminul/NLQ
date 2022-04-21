python evaluate_ego4d_nlq.py \
--ground_truth_json /playpen-storage/mmiemon/ego4d/data/annotations/nlq_val.json \
--model_prediction_json results/all_data_mean_pool/ego4d-video_tef-queries_20/ego4d_nlq_val.json \
--thresholds 0.1 0.3 0.5 \
--topK 1 3 5\