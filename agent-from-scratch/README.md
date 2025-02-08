>内容：
>
>agent各个组件简易实现

## 环境

```ps
conda create -n agent python=3.10
conda activate agent
pip install openai
pip install pytz
```

需要额外开个环境起一个模型来适配openai格式。

## 运行

run_agent.ipynb

## some

这个是写完DesktopDoraemon之后[一气呵成/灵机一动/...省略若干成语]搞的，没有很多依赖，没有很精心设计prompt，当然也没有很多代码……需要脑子转一下的是对llm返回结果的处理：如果能执行动作，执行，然后把动作、参数、工具返回追加到user方的消息，再让llm继续生成。