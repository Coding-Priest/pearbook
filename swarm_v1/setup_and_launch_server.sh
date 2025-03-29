# You MUST replace <PEER_ID_OF_NODE_ON_45651> with the actual Peer ID
# from the logs of the process running on localhost:45651

python -m hivemind.hivemind_cli.run_server \
--num_experts 1 \
--expert_pattern head.0.[0:127] --expert_cls lm_head --hidden_dim 4096 --num_handlers 64 \
--scheduler linear --fp16 --stats_report_interval 60 \
--num_warmup_steps 3125 --num_total_steps 15000 --clip_grad_norm 1.0 --compression BLOCKWISE_8BIT \
--averaging_target_batch_size 4096 --averaging_expiration 60 --averaging_timeout 700 --metadata_expiration 700 \
--min_batch_size 1 --max_batch_size 1 --offload \
--device cuda:0 \
--listen_on 127.0.0.1:* \
--dht_listen_on ip4/127.0.0.1 \
--initial_peers '/ip4/127.0.0.1/tcp/40211/p2p/12D3KooWKHcAzHbCyaSnKbtf33sLSNACxvJCJheDR7qXsJtkKoDy' \
2>&1 | tee -a server_stderr_head_0.log