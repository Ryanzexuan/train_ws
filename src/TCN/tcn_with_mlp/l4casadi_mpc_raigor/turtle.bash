#!/bin/zsh

# 启动 turtlesim 节点
rosrun turtlesim turtlesim_node __name:=my_turtle &

# 等待 turtlesim 启动
while ! rostopic list | grep -q /my_turtle; do
    sleep 1
done

# 调用 teleport 服务设置初始位置
rosservice call /my_turtle/teleport_absolute 0.0 0.0 0.0
