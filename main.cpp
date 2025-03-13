#include <iostream>
#include <filesystem>
#include <future>
#include <vector>
#include <string>
#include <expected>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

#define var auto
#define fa(i,op,n) for(int i=op;i<=n;i++)
#define fb(i,op,n) for(int i=op;i>=n;i--)
#define UNEPECTED_FUNC(error) std::unexpected(std::string(__func__) + "：" + error)


namespace fs = std::filesystem;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

template<typename Func>
class TimeMeasurerClass {
	// - TimeMeasurer 函数对象
	// - 函数计时器
	// - 好像不能当右值用？不能 TMC(Func)(Args)
public:
	// explicit 单参数构造函数中使用，避免意外的隐式类型转换
	explicit TimeMeasurerClass(Func func) :func(func) {};

	template<typename ...Args>
	auto operator()(Args&&... args) {
		// Args&&...万能引用传入

		auto start = std::chrono::steady_clock::now();
		auto result = func(std::forward<Args>(args)...); // 保证参数类型不变，然后逗号展开
		auto end = std::chrono::steady_clock::now();
		std::cout << "耗时: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms" << std::endl;
		return result;
	}
private:
	Func func;
};

template<typename Func, typename ...Args>
auto TimeMeasurerMethod(Func&& func, Args&&... args) {
	auto start = std::chrono::steady_clock::now();
	auto result = func(std::forward<Args>(args)...); // 保证参数类型不变，然后逗号展开
	auto end = std::chrono::steady_clock::now();
	std::cout << "耗时: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< "ms" << std::endl;
	return result;
}


cudaError_t gpu_down_sample(cv::Mat& old_image, std::shared_ptr<uchar[]> new_image);


fs::directory_iterator get_paths();
std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>async_imread(const fs::directory_iterator& list);


std::expected<void, std::string>test_opencv();
std::expected<void, std::string>test_filesystem();
std::expected<void, std::string>test_sync_imread(const fs::directory_iterator& list);
std::expected<void, std::string>test_opencv_down_sample();
std::expected<void, std::string>test_cpu_down_sample();
std::expected<void, std::string>test_gpu_down_sample();
std::expected<void, std::string>test_imread_then_cpu_down_sample();
std::expected<void, std::string>test_imread_then_gpu_down_sample_without_cudaStream();
std::expected<void, std::string>test_imread_then_opencv_down_sample();
std::expected<void, std::string>test_opencv_down_quality();



int main()
{
	TimeMeasurerMethod(test_imread_then_cpu_down_sample);
	return 0;
}


std::expected<void, std::string>
test_imread_then_gpu_down_sample_without_cudaStream() {
	/**
	* 异步 imread 图片后，在不使用 Cuda 流的情况下使用 std::async 调用核函数降低图像分辨率
	*/

	var images = vector<std::pair<std::string, cv::Mat>>();
	async_imread(fs::directory_iterator(fs::path("./images")))

		.or_else([](var err)
			->std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>
			{std::cerr << err << endl; return vector<std::pair<std::string, cv::Mat>>(); })

		.and_then([&images](var res)
			->std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>
			{ images = std::move(res); return {}; });
		

	var futures = vector<std::future<std::expected<void, std::string>>>();
	for (var& image : images) {
		futures.push_back(std::async(std::launch::async, 
			[&image]()->std::expected<void, std::string> {
				var& [name, ori_image] = image;
				int rows = ori_image.rows;
				int cols = ori_image.cols;
				var new_image = std::make_shared<uchar[]>(rows / 2 * cols / 2 * 3);

				if (gpu_down_sample(ori_image, new_image) != cudaSuccess)
					return UNEPECTED_FUNC("CUDA处理失败");

				var success = cv::imwrite(
					"./output/" + name + ".jpg",
					cv::Mat(rows / 2, cols / 2, CV_8UC3, new_image.get())
				);
				if (!success)return UNEPECTED_FUNC("图像生成失败");

				return {};
			}));
	}

	for (var& future : futures) {
		var now = future.get();
		if (!now.has_value()) {
			return UNEPECTED_FUNC(now.error());
		}
	}
	return {};
}

std::expected<void, std::string>
test_imread_then_cpu_down_sample() {
	/**
	* 异步 imread 图片后，使用纯 CPU 和 std::async 降低图像分辨率
	*/

	var images = vector<std::pair<std::string, cv::Mat>>();
	async_imread(fs::directory_iterator(fs::path("./images")))

		.or_else([](var err)
			->std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>
			{std::cerr << err << endl; return vector<std::pair<std::string, cv::Mat>>(); })

		.and_then([&images](var res)
			->std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>
			{ images = std::move(res); return {}; });


	var futures = vector<std::future<std::expected<void, std::string>>>();
	for (var& image : images) {
		futures.push_back(std::async(std::launch::async,
			[&image]()->std::expected<void, std::string> {
				var& [name, ori_image] = image;
				int n = ori_image.rows;
				int m = ori_image.cols;
				var new_image = cv::Mat(n / 2, m / 2, ori_image.type());
				int new_n = n / 2;
				int new_m = m / 2;

				fa(x, 0, new_n - 1)
					fa(y, 0, new_m - 1) {
					int ori_x = x * 2;
					int ori_y = y * 2;
					fa(c, 0, 2) {
						uchar sum = 0;
						fa(dx, 0, 1)
							fa(dy, 0, 1) {
							if (ori_x + dx >= n or ori_y + dy >= m)continue;
							int nx = ori_x + dx, ny = ori_y + dy;

							sum += ori_image.data[nx * m * 3 + ny * 3 + c] / (2 * 2);
						}
						new_image.data[x * new_m * 3 + y * 3 + c] = sum;
					}
				}

				var success = cv::imwrite(
					"./output/" + name + ".jpg",
					new_image
				);
				if (!success)return UNEPECTED_FUNC("图像生成失败");

				return {};
			}));
	}

	for (var& future : futures) {
		var now = future.get();
		if (!now.has_value()) {
			return UNEPECTED_FUNC(now.error());
		}
	}
	return {};
}

std::expected<void, std::string>
test_imread_then_opencv_down_sample() {
	/**
	* 异步 imread 图片后，使用纯 CPU 和 std::async 降低图像分辨率
	*/

	var images = vector<std::pair<std::string, cv::Mat>>();
	async_imread(fs::directory_iterator(fs::path("./images")))

		.or_else([](var err)
			->std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>
			{std::cerr << err << endl; return vector<std::pair<std::string, cv::Mat>>(); })

		.and_then([&images](var res)
			->std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>
			{ images = std::move(res); return {}; });


	var futures = vector<std::future<std::expected<void, std::string>>>();
	for (var& image : images) {
		futures.push_back(std::async(std::launch::async,
			[&image]()->std::expected<void, std::string> {
				var& [name, ori_image] = image;
				int new_rows = ori_image.rows / 2;
				int new_cols = ori_image.cols / 2;

				cv::Mat new_image;
				cv::resize(ori_image, new_image, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_AREA);

				var success = cv::imwrite(
					"./output/" + name + ".jpg",
					new_image
				);
				if (!success)return UNEPECTED_FUNC("图像生成失败");

				return {};
			}));
	}

	for (var& future : futures) {
		var now = future.get();
		if (!now.has_value()) {
			return UNEPECTED_FUNC(now.error());
		}
	}
	return {};
}


std::expected<void, std::string>
test_gpu_down_sample() {
	var old_image = cv::imread("test.jpg");
	int rows = old_image.rows;
	int cols = old_image.cols;
	var new_image = std::make_shared<uchar[]>(rows / 2 * cols / 2 * 3);

	if (gpu_down_sample(old_image, new_image) != cudaSuccess)
		return UNEPECTED_FUNC("CUDA处理失败");

	var success = cv::imwrite("test_output.jpg", cv::Mat(rows / 2, cols / 2, CV_8UC3, new_image.get()));
	if (!success)return UNEPECTED_FUNC("图像生成失败");

	return {};
}

std::expected<void, std::string>
test_cpu_down_sample() {
	var ori_image = cv::imread("test.jpg");
	int n = ori_image.rows;
	int m = ori_image.cols;
	var image = cv::Mat(n / 2, m / 2, ori_image.type());
	int new_n = n / 2;
	int new_m = m / 2;
	fa(x, 0, new_n - 1)
		fa(y, 0, new_m - 1) {
		int ori_x = x * 2;
		int ori_y = y * 2;
		fa(c, 0, 2) {
			uchar sum = 0;
			fa(dx, 0, 1)
				fa(dy, 0, 1) {
				if (ori_x + dx >= n or ori_y + dy >= m)continue;
				int nx = ori_x + dx, ny = ori_y + dy;

				sum += ori_image.data[nx * m * 3 + ny * 3 + c] / (2 * 2);
			}
			image.data[x * new_m * 3 + y * 3 + c] = sum;
		}
	}
	var success = cv::imwrite("test_output.jpg", image);
	if (!success)return UNEPECTED_FUNC("图像生成失败");

	return {};
}


std::expected<void, std::string>
test_opencv_down_sample() {
	var old_image = cv::imread("test.jpg");
	int new_rows = old_image.rows / 2;
	int new_cols = old_image.cols / 2;

	cv::Mat new_image;
	cv::resize(old_image, new_image, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_AREA);
	cv::imwrite("output_test.jpg", new_image);
	return {};
}

std::expected<void, std::string>
test_opencv_down_quality() {
	var image = cv::imread("test.jpg");
	bool success = cv::imwrite("output_test.jpg", image, { cv::IMWRITE_JPEG_QUALITY, 50 });
	return {};
}


std::expected<void, std::string>
test_sync_imread(const fs::directory_iterator& list) {
	/**
	* 同步 imread
	*/
	
	for (const var& it : list) {
		cv::Mat img = cv::imread(it.path().string());
	}
	return {};
}

std::expected<void, std::string>
test_opencv() {
	cv::Mat test = cv::imread("test.png");
	if (test.empty()) {
		return UNEPECTED_FUNC("读取图片失败");
	}
	cv::namedWindow("test", cv::WINDOW_FREERATIO);
	cv::imshow("test", test);
	cv::waitKey(0);
	return {};
}

std::expected<void, std::string>
test_filesystem() {
	fs::path path("./images");
	cv::imread(path.string());
	fs::directory_entry entry(path);
	if (entry.is_directory()) {
		fs::directory_iterator list(path);
		for (const var& it : list) {
			var str = it.path().string();
			std::replace(str.begin(), str.end(), '\\', '/');
			cout << str << endl;
		}
	}
	return {};
}


std::expected<vector<std::pair<std::string, cv::Mat>>, std::string>
async_imread(const fs::directory_iterator& list) {
	/**
	* 使用 async 自动创建线程进行异步 imread
	*	- async 策略，新建线程并在线程内立即执行函数
	*	- defer 策略，延迟并在 get() 函数调用时同步执行函数
	*   - async|deferred 默认策略，线程有空闲则执行
	*/

	var futures = vector<std::future<std::pair<std::string, cv::Mat>>>();

	for (const var& it : list) {
		futures.push_back(std::async(std::launch::async,
			[it]() { // 循环结束后引用会自动销毁，得使用拷贝 (涉及 directory_iterator 的底层实现)
				var str = it.path().string();
				std::replace(str.begin(), str.end(), '\\', '/');

				return std::pair<std::string, cv::Mat>{
					it.path().stem().string(), cv::imread(str)
				};
			}));
	}

	var res = vector<std::pair<std::string, cv::Mat>>();
	for (var& future : futures) {
		res.push_back(future.get());
	}
	return res;
}

fs::directory_iterator
get_paths() {
	fs::path path("./images");
	fs::directory_iterator list(path);
	return list;
}



/* 
1. uchar 数组转换 Mat 注意事项：
	- 不能直接用 vector 多维数组，内存不连续
	var ve = vector<vector<uchar>>(n, vector<uchar>(m));
	var image = cv::Mat(n, m, CV_8UC1, ve.data);

	- 只能用单维 vector，多维 vector 作辅助，或者直接用 uchar ve[][]
	var ve = vector<vector<uchar>>(n, vector<uchar>(m));
	var target = vector<uchar>();
	for (const var&i : ve) {
		target.insert(target.end(), i.begin(), i.end());
	}
*/