ps -ef | grep "dialogbot_gunicorn_config.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9
#nohup python3.6 web/server.py >> log.txt 2>&1 &
cd /home/timo/web/dialogbot/dialogbot/web
gunicorn -c dialogbot_gunicorn_config.py server:app
