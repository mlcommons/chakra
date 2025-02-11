# latest news
截至11.17，同步了chakra的最新版本

# chakra-dts
模型表达提取工具

chakra_cleaner用于对ET计算图文件进行剪枝

chakra_jsonizer用于将ET计算图json格式化

chakra_timeline_visualizer可以将ET计算图以时间线的方式可视化

chakra_converter用于将pytorch ET plus文件转换为标准ET文件

chakra_parallel_expander用于将当前的ET计算图进行扩展

chakra_trace_link用于将pytorch ET和kineto trace信息合并，合并后的文件我们称为pytorch ET plus

chakra_generator可用于生成一些测试用例

chakra_pg_extractor用于提取通信域描述文件

chakra_visualizer用于将ET计算图可视化为dot文件，该dot文件可以用graphize online在线查看

工具用法的命令可以用-h进行查看

## Installation
```python
pip install .
```

## Usage
[`quick_process.sh`](quick_process.sh)
```bash
bash quick_process.sh
```

## License

Chakra is released under the MIT license. Please see the [`LICENSE.md`](LICENSE.md) file for more information.

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for more info.
