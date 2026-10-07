# E2 OS / WS / IS

OS：输出部分和跨K驻留；每M波次从SRAM重读W。WS：W驻留依次服务M，K段部分和在累加SRAM读改写。IS：X驻留遍历N，部分和读改写；组会RS在此等同IS。


pipelined（fixed_issue为非等资源参考）：
- B1: 最快WS/-；OS/OS比最快慢58.23%。
- fixed_4+2: 最快WS/WS；OS/OS比最快慢26.22%。
- previous_asym: 最快WS/IS；OS/OS比最快慢23.51%。
- best_hetero: 最快WS/WS；OS/OS比最快慢107.93%。

port_tight（fixed_issue为非等资源参考）：
- B1: 最快WS/-；OS/OS比最快慢58.76%。
- fixed_4+2: 最快WS/WS；OS/OS比最快慢46.35%。
- previous_asym: 最快WS/OS；OS/OS比最快慢14.88%。
- best_hetero: 最快WS/WS；OS/OS比最快慢109.02%。

fixed_issue（fixed_issue为非等资源参考）：
- B1: 最快WS/-；OS/OS比最快慢55.98%。
- fixed_4+2: 最快WS/WS；OS/OS比最快慢26.78%。
- previous_asym: 最快WS/OS；OS/OS比最快慢9.48%。
- best_hetero: 最快WS/WS；OS/OS比最快慢25.73%。

Me≤PM在pipelined下OS与WS的时间和全部流量均相同：0/42；不得把一次M发射等同于全部数据流相同。

pipelined/routed多M波次微实验：WS严格快于OS 42/45 点；配对几何平均WS/OS=0.505818。逐Me与形状差异见micro_WS_vs_OS.csv，不能把此均值等同整层加速。

pipelined/Shared多M波次微实验：WS严格快于OS 42/45 点；配对几何平均WS/OS=0.505026。逐Me与形状差异见micro_WS_vs_OS.csv，不能把此均值等同整层加速。

pipelined/fixed_4+2: OS/OS与最快流的延迟比=1.262192，不在最快值1%以内。

pipelined/previous_asym: OS/OS与最快流的延迟比=1.235144，不在最快值1%以内。

pipelined/best_hetero: OS/OS与最快流的延迟比=2.079270，不在最快值1%以内。

Me≤PM在port_tight下OS与WS的时间和全部流量均相同：0/42；不得把一次M发射等同于全部数据流相同。

port_tight/routed多M波次微实验：WS严格快于OS 45/45 点；配对几何平均WS/OS=0.256956。逐Me与形状差异见micro_WS_vs_OS.csv，不能把此均值等同整层加速。

port_tight/Shared多M波次微实验：WS严格快于OS 45/45 点；配对几何平均WS/OS=0.258306。逐Me与形状差异见micro_WS_vs_OS.csv，不能把此均值等同整层加速。

port_tight/fixed_4+2: OS/OS与最快流的延迟比=1.463474，不在最快值1%以内。

port_tight/previous_asym: OS/OS与最快流的延迟比=1.148754，不在最快值1%以内。

port_tight/best_hetero: OS/OS与最快流的延迟比=2.090171，不在最快值1%以内。

Me≤PM在fixed_issue下OS与WS的时间和全部流量均相同：0/42；不得把一次M发射等同于全部数据流相同。

fixed_issue/routed多M波次微实验：WS严格快于OS 35/45 点；配对几何平均WS/OS=0.937847。逐Me与形状差异见micro_WS_vs_OS.csv，不能把此均值等同整层加速。

fixed_issue/Shared多M波次微实验：WS严格快于OS 35/45 点；配对几何平均WS/OS=0.937190。逐Me与形状差异见micro_WS_vs_OS.csv，不能把此均值等同整层加速。

fixed_issue/fixed_4+2: OS/OS与最快流的延迟比=1.267779，不在最快值1%以内。

fixed_issue/previous_asym: OS/OS与最快流的延迟比=1.094823，不在最快值1%以内。

fixed_issue/best_hetero: OS/OS与最快流的延迟比=1.257270，不在最快值1%以内。

