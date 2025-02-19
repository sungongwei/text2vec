# 关闭程序
fileName=openai:app
pid=$(ps -ef | grep $fileName| grep -v "grep" | awk '{print $2}')
kill -9 $pid

# 启动项目
gunicorn --workers 8 openai:app &
