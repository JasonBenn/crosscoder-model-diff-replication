copy_nginx:
	cp /workspace/crosscoder-model-diff-replication/nginx-content.conf /etc/nginx/sites-enabled/crosscoder.conf
	nginx -s reload
