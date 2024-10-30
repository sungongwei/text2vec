# 关闭程序
# fileName为jar包的名称
fileName=api.py
pid=$(ps -ef | grep $fileName| grep -v "grep" | awk '{print $2}')
kill -9 $pid

# 启动项目
nohup python $fileName  &
