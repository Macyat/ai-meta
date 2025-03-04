import datetime
import json
import threading
import socket
import time
import tkinter as tk
from functools import partial, cache
from tkinter import messagebox, ttk, RAISED
from tkinter.ttk import Style

# from PIL import Image, ImageTk
from loguru import logger

import full_service
import full_service_blind
import full_service_standard
import init
import service

# import service
from common.enums import ActionEnum
from common.enums import OpticalLengthEnum
from common.enums import TypeEnum
from common.enums import ExamType, TurExamType

# from util.log import logger
from control import ctrl
from sensor.cod import CODSensor
from sensor.ph import PHSensor
from spectrograph import spectrograph
from task import task
from util import util, signal_client
from util.ini import IniFile
from sensor.rain import RainSensor

log_file = f"logs/water-{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
logger.add(log_file, level="DEBUG", rotation="1 day", retention="15 days")

stop_event = threading.Event()
stop_event.set()
digest_event = threading.Event()
digest_event.set()


def update_weather():
    while True:
        try:
            rain = RainSensor()
            cumulative_rain, instantaneous_rain = rain.read_Rain_data()
            rain.close()
            time.sleep(60 * 10)
        except Exception as err:
            logger.error("---更新天气数据库异常---{}", err)


class MainFrame(tk.Tk):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.title("宏光谱在线水质检测仪")
        # self.iconphoto(True, tk.PhotoImage(file='light.png'))
        self.ini = IniFile("config/config.ini")
        self.ctrl = ctrl()
        self.spectrograph = spectrograph()
        self.task = task(self)
        self.task.add_status_job()
        # self.task.add_data_job()
        self.task.add_shake_hand_job()
        self.task.add_init_job()
        self.task.add_service_job()
        self.task.start()
        signal_client.report_open()

        # self.geometry("1000x600")
        # center_window(self, 1100, 600)
        self.state("zoomed")  # 设置窗口状态为最大化

        # 创建菜单
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # 创建菜单项
        parameter_control_menu = tk.Menu(menubar, tearoff=0)
        parameter_control_menu.add_command(
            label="参数设置", command=self.open_parameter_control
        )
        parameter_control_menu.add_command(
            label="定时计划", command=self.open_timer_control
        )
        menubar.add_cascade(label="文件", menu=parameter_control_menu)
        parameter_control_menu.add_command(label="退出", command=self.on_close)

        control_unit_menu = tk.Menu(menubar, tearoff=0)
        control_unit_menu.add_command(
            label="半自动控制", command=self.open_control_unit
        )
        control_unit_menu.add_command(
            label="IO监控与点动控制", command=self.open_device_control
        )
        menubar.add_cascade(label="控制", menu=control_unit_menu)

        tools_unit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="测试工具", menu=tools_unit_menu)

        tn_menu = tk.Menu(tools_unit_menu, tearoff=0)
        tools_unit_menu.add_cascade(label="总氮", menu=tn_menu)
        tn_menu.add_command(
            label="漂移", command=lambda: self.test_drift("DRIFT", "漂移", "TN")
        )
        tn_menu.add_command(
            label="定量下限",
            command=lambda: self.test_lowerlimit("LOWER_LIMIT", "定量下限", "TN"),
        )
        tn_menu.add_command(
            label="示值误差",
            command=lambda: self.test_indicationerror(
                "INDICATION_ERROR", "示值误差", "TN"
            ),
        )
        tn_menu.add_command(
            label="重复性",
            command=lambda: self.test_repeatability("REPEATABILITY", "重复性", "TN"),
        )
        tn_menu.add_command(
            label="一致性偏差",
            command=lambda: self.test_consistency("CONSISTENCY", "一致性偏差", "TN"),
        )

        cod_menu = tk.Menu(tools_unit_menu, tearoff=0)
        tools_unit_menu.add_cascade(label="COD", menu=cod_menu)
        cod_menu.add_command(
            label="漂移", command=lambda: self.test_drift("DRIFT", "漂移", "COD")
        )
        cod_menu.add_command(
            label="定量下限",
            command=lambda: self.test_lowerlimit("LOWER_LIMIT", "定量下限", "COD"),
        )
        cod_menu.add_command(
            label="示值误差",
            command=lambda: self.test_indicationerror(
                "INDICATION_ERROR", "示值误差", "COD"
            ),
        )
        cod_menu.add_command(
            label="重复性",
            command=lambda: self.test_repeatability("REPEATABILITY", "重复性", "COD"),
        )
        cod_menu.add_command(
            label="一致性偏差",
            command=lambda: self.test_consistency("CONSISTENCY", "一致性偏差", "COD"),
        )

        kmno_menu = tk.Menu(tools_unit_menu, tearoff=0)
        tools_unit_menu.add_cascade(label="高锰酸盐", menu=kmno_menu)
        kmno_menu.add_command(
            label="漂移", command=lambda: self.test_drift("DRIFT", "漂移", "KMNO")
        )
        kmno_menu.add_command(
            label="定量下限",
            command=lambda: self.test_lowerlimit("LOWER_LIMIT", "定量下限", "KMNO"),
        )
        kmno_menu.add_command(
            label="示值误差",
            command=lambda: self.test_indicationerror(
                "INDICATION_ERROR", "示值误差", "KMNO"
            ),
        )
        kmno_menu.add_command(
            label="重复性",
            command=lambda: self.test_repeatability("REPEATABILITY", "重复性", "KMNO"),
        )
        kmno_menu.add_command(
            label="一致性偏差",
            command=lambda: self.test_consistency("CONSISTENCY", "一致性偏差", "KMNO"),
        )

        tur_menu = tk.Menu(tools_unit_menu, tearoff=0)
        tools_unit_menu.add_cascade(label="浊度", menu=tur_menu)
        tur_menu.add_command(
            label="漂移",
            command=lambda: self.test_drift("DRIFT", "漂移", "TURBIDITY", True),
        )
        tur_menu.add_command(
            label="示值误差",
            command=lambda: self.test_indicationerror(
                "INDICATION_ERROR", "示值误差", "TURBIDITY", True
            ),
        )
        tur_menu.add_command(
            label="重复性",
            command=lambda: self.test_repeatability(
                "REPEATABILITY", "重复性", "TURBIDITY", True
            ),
        )

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="关于", command=self.open_about)
        menubar.add_cascade(label="帮助", menu=about_menu)

        # 创建初始化和检测按钮
        # init_button = tk.Button(self, text="初始化", width=40, height=15, command=self.on_initialize)
        # init_button.pack(pady=10)
        #
        # check_button = tk.Button(self, text="检测自动运行", width=40, height=15, command=self.on_check)
        # check_button.pack(pady=5)
        #
        # stop_button = tk.Button(self, text="关闭所有设备", width=20, height=15, command=self.on_close_ctrl)
        # stop_button.pack(side='right', anchor='s', padx=10, pady=1)

        # Load the image
        # img = Image.open('1920.jpg')
        # # Resize the image to fit the window size
        # img = img.resize((self.winfo_screenwidth(), self.winfo_screenheight()), Image.ANTIALIAS)
        # # Create a PhotoImage object from the image
        # photo = ImageTk.PhotoImage(img)
        # # Create a label to hold the image
        # label_bg = tk.Label(self, image=photo)
        # label_bg.image = photo  # This line is necessary to prevent the image from being garbage collected
        # label_bg.place(x=0, y=0, relwidth=1, relheight=1)

        # 在frame_1上添加另外两个frame， 一个在靠上，一个靠下
        # 上侧的frame
        frame_top = tk.Frame(self, width=1000, height=20)
        self.label_time = tk.Label(frame_top, text="时间", font=40, width=20, height=5)
        self.label_time.grid(row=0, column=0)
        self.label_signal = tk.Label(
            frame_top, text="信号", font=40, width=10, height=5
        )
        self.label_signal.grid(row=0, column=1)
        self.label_eltrc_amount = tk.Label(
            frame_top, text="电量", font=40, width=10, height=5
        )
        self.label_eltrc_amount.grid(row=0, column=2)
        frame_top.pack(side=tk.TOP)
        frame_bottom = tk.Frame(self, width=1000, height=700)
        frame_bottom.pack(side=tk.TOP, expand=True, fill="both")

        # 样式
        style1 = Style()
        style1.configure(
            "TNotebook", tabposition="s", tabwidth=1000
        )  # 'se'再改nw,ne,sw,se,w,e,wn,ws,en,es,n,s试试
        style1.configure("TNotebook.Tab", font=("黑体", 22, "bold"))
        # 设置选项卡字体，大小，颜色
        # 创建 Notebook（标签页容器）
        notebook = ttk.Notebook(frame_bottom, style="TNotebook")
        notebook.pack(side=tk.TOP, expand=True, fill="both")
        # 创建第一个标签页
        tab1 = ttk.Frame(notebook, borderwidth=1, padding=(20, 20, 20, 20))
        notebook.add(tab1, text="  首页  ")

        #
        self.frame_left = tk.LabelFrame(
            tab1, text="运行状态", font=(22), width=650, height=700
        )
        self.frame_left.pack(side=tk.LEFT, padx=(10, 10), pady=10, fill="both")
        self.label_drain_sonic = tk.Label(self.frame_left, text="消解超声", font=30)
        self.label_drain_sonic.grid(row=0, column=0, pady=20, sticky="W")
        self.label_water_empty_water1 = tk.Label(
            self.frame_left, text="消解池排空", font=30
        )
        self.label_water_empty_water1.grid(row=0, column=1, pady=20, sticky="W")
        self.label_eltrc = tk.Label(self.frame_left, text="电解", font=30)
        self.label_eltrc.grid(row=0, column=2, pady=20, sticky="W")
        self.label_detect_sonic1 = tk.Label(self.frame_left, text="检测-超声", font=30)
        self.label_detect_sonic1.grid(row=0, column=3, pady=20, sticky="W")

        self.label_empty_check_water = tk.Label(
            self.frame_left, text="检测池排空", font=30
        )
        self.label_empty_check_water.grid(row=1, column=0, pady=20, sticky="W")
        self.label_empty = tk.Label(self.frame_left, text="原水排空", font=30)
        self.label_empty.grid(row=1, column=1, pady=20, sticky="W")
        self.label_sample = tk.Label(self.frame_left, text="采样", font=30)
        self.label_sample.grid(row=1, column=2, pady=20, sticky="W")
        self.label_water_sonic1 = tk.Label(self.frame_left, text="原水超声", font=30)
        self.label_water_sonic1.grid(row=1, column=3, pady=20, sticky="W")
        self.label_water_stand = tk.Label(self.frame_left, text="原水静置", font=30)

        self.label_water_stand.grid(row=2, column=0, pady=20, sticky="W")
        self.label_sample_water_pool = tk.Label(
            self.frame_left, text="检测池加样", font=30
        )
        self.label_sample_water_pool.grid(row=2, column=1, pady=20, sticky="W")
        self.label_detect_sonic2 = tk.Label(self.frame_left, text="检测超声", font=30)
        self.label_detect_sonic2.grid(row=2, column=2, pady=20, sticky="W")
        self.label_spectrograph = tk.Label(
            self.frame_left, text="消解前光谱测量", font=30
        )
        self.label_spectrograph.grid(row=2, column=3, pady=20, sticky="W")

        self.label_water_ctrl_water = tk.Label(
            self.frame_left, text="加样-检测-消解池", font=30
        )
        self.label_water_ctrl_water.grid(row=3, column=0, pady=20, sticky="W")
        self.label_drain_process = tk.Label(self.frame_left, text="消解流程", font=30)
        self.label_drain_process.grid(row=3, column=1, pady=20, sticky="W")
        self.label_drain_remove_O3 = tk.Label(
            self.frame_left, text="消解去臭氧", font=30
        )
        self.label_drain_remove_O3.grid(row=3, column=2, pady=20, sticky="W")
        self.label_drain_stand = tk.Label(self.frame_left, text="消解静置", font=30)
        self.label_drain_stand.grid(row=3, column=3, pady=20, sticky="W")

        self.label_water_empty_water2 = tk.Label(
            self.frame_left, text="消解池排空", font=30
        )
        self.label_water_empty_water2.grid(row=4, column=0, pady=20, sticky="W")
        self.label_water_sonic2 = tk.Label(self.frame_left, text="检测-超声", font=30)
        self.label_water_sonic2.grid(row=4, column=1, pady=20, sticky="W")
        self.label_spectrograph_uv = tk.Label(
            self.frame_left, text="消解后光谱测量", font=30
        )
        self.label_spectrograph_uv.grid(row=4, column=2, pady=20, sticky="W")

        frame_right = tk.LabelFrame(tab1, text="数据", font=(22), width=600, height=700)
        frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill="both")
        self.label_tp = tk.Label(frame_right, text="总磷", font=30, width=15, height=5)
        self.label_tp.grid(row=0, column=0, sticky="W")
        self.label_tn = tk.Label(frame_right, text="总氮", font=30, width=15, height=5)
        self.label_tn.grid(row=0, column=1, sticky="W")
        self.label_cod = tk.Label(frame_right, text="COD", font=30, width=15, height=5)
        self.label_cod.grid(row=0, column=2, sticky="W")
        self.label_an = tk.Label(frame_right, text="氨氮", font=30, width=15, height=5)
        self.label_an.grid(row=3, column=0, sticky="W")
        self.label_do = tk.Label(
            frame_right, text="溶解氧", font=30, width=15, height=5
        )
        self.label_do.grid(row=3, column=1, sticky="W")
        self.label_conduct = tk.Label(
            frame_right, text="电导率", font=30, width=15, height=5
        )
        self.label_conduct.grid(row=3, column=2, sticky="W")
        self.label_ph = tk.Label(frame_right, text="PH", font=30, width=15, height=5)
        self.label_ph.grid(row=5, column=0, sticky="W")
        self.label_temp = tk.Label(
            frame_right, text="温度", font=30, width=15, height=5
        )
        self.label_temp.grid(row=5, column=1, sticky="W")
        self.label_tur = tk.Label(frame_right, text="浊度", font=30, width=15, height=5)
        self.label_tur.grid(row=5, column=2, sticky="W")
        # 创建第二个标签页
        tab2 = ttk.Frame(notebook, borderwidth=1, padding=(20, 20, 20, 20))
        notebook.add(tab2, text="  设置  ")

        # 在第二个标签页中添加内容
        timer_button = tk.Button(
            tab2,
            text="定时计划",
            font=(40),
            width=50,
            height=300,
            command=self.open_timer_control,
        )
        timer_button.pack(side="left", anchor="e", padx=(100, 10), pady=1)

        params_button = tk.Button(
            tab2,
            text="参数设置",
            font=(40),
            width=50,
            height=300,
            command=self.open_parameter_control,
        )
        params_button.pack(side="right", anchor="w", padx=(10, 200), pady=1)

        tab3 = ttk.Frame(notebook, borderwidth=1, padding=(20, 20, 20, 20))
        notebook.add(tab3, text="  调试  ")

        io_button = tk.Button(
            tab3,
            text="IO监控与点动控制",
            font=(40),
            width=50,
            height=10,
            command=self.open_device_control,
        )
        io_button.pack(side="top", anchor="n", padx=1, pady=10)

        ctrl_button = tk.Button(
            tab3,
            text="半自动控制按钮",
            font=(40),
            width=50,
            height=10,
            command=self.open_control_unit,
        )
        ctrl_button.pack(side="bottom", anchor="s", padx=1, pady=10)

        # tab4 = ttk.Frame(notebook,borderwidth=1, padding=(20,20,20,20))
        # notebook.add(tab4, text="  数据  ")

        # 设置主窗口的protocol属性，将WM_DELETE_WINDOW事件绑定到主窗口的关闭操作
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_time()
        self.update_data()
        # 启动线程
        # t1 = threading.Thread(target=update_weather)
        # t1.start()

        (
            self.X_check_vars,
            self.X_1_check_vars,
            self.Y_check_vars,
            self.Y_1_check_vars,
        ) = (None, None, None, None)
        self.init_hours, self.service_hours = None, None
        (
            self.entry_score,
            self.entry_avg,
            self.entry_high,
            self.entry_low,
            self.entry_sample,
            self.entry_drain,
        ) = (None, None, None, None, None, None)

    def update_time(self):
        if "lyz" == socket.gethostname():
            return
        # 获取当前时间
        now = datetime.datetime.now()

        # 格式化时间
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.label_time.config(text=formatted_time)
        amount = 100
        try:
            amount = self.ctrl.get_battery_router()
        except Exception as e:
            logger.error(e)
        finally:
            self.label_eltrc_amount.config(text="电量:" + str(amount) + "%")
            logger.info("UI状态已更新time={}", formatted_time)

    def update_data(self):
        try:
            data = self.ctrl.get_lastest_data()

            self.label_an.config(text="氨氮:" + str(data["data"]["predictedAn"]))
            self.label_tp.config(text="总磷:" + str(data["data"]["predictedTp"]))
            self.label_tn.config(text="总氮:" + str(data["data"]["predictedTn"]))
            self.label_cod.config(text="COD:" + str(data["data"]["predictedCod"]))
            self.label_ph.config(text="PH:" + str(data["data"]["ph"]))
            self.label_do.config(text="溶解氧:" + str(data["data"]["dissolvedOxygen"]))
            self.label_temp.config(text="温度:" + str(data["data"]["temperature"]))
            self.label_conduct.config(
                text="电导率:" + str(data["data"]["conductivity"])
            )
            self.label_tur.config(text="浊度:" + str(data["data"]["turbidity"]))

            logger.info("data UI状态已更新")
        except Exception as e:
            logger.error(e)
        finally:
            logger.info("data UI状态结束")

    def open_parameter_control(self):
        # 这里可以添加打开参数控制界面的代码

        # 这里可以添加打开控制单元界面的代码
        # 创建新的Toplevel窗口
        parameter_window = tk.Toplevel(self)
        # 使窗口变为模态
        # parameter_window.transient(self)
        # parameter_window.grab_set()
        # button1.grid(row=10, column=5, padx=5, pady=5)

        # 设置窗口标题和大小
        parameter_window.title("参数")
        # parameter_window.geometry("600x400")
        center_window(parameter_window, 850, 500)

        label_score = tk.Label(parameter_window, text=f"光谱仪积分")
        label_score.grid(row=2, column=0, padx=5, pady=50)

        self.entry_score = tk.Entry(parameter_window)
        self.entry_score.grid(row=2, column=1, padx=5, pady=50)
        score = self.ini.get_option("spectrograph", "score")
        self.entry_score.insert(0, score)

        label_avg = tk.Label(parameter_window, text=f"平均时间")
        label_avg.grid(row=2, column=2, padx=5, pady=50)

        self.entry_avg = tk.Entry(parameter_window)
        self.entry_avg.grid(row=2, column=3, padx=5, pady=50)
        average_time = self.ini.get_option("spectrograph", "average_time")
        self.entry_avg.insert(0, average_time)

        label_high = tk.Label(parameter_window, text=f"高电平")
        label_high.grid(row=2, column=4, padx=5, pady=50)

        self.entry_high = tk.Entry(parameter_window)
        self.entry_high.grid(row=2, column=5, padx=5, pady=50)
        high_level = self.ini.get_option("spectrograph", "high_level")
        self.entry_high.insert(0, high_level)

        label_low = tk.Label(parameter_window, text=f"低电平")
        label_low.grid(row=2, column=6, padx=5, pady=50)

        self.entry_low = tk.Entry(parameter_window)
        self.entry_low.grid(row=2, column=7, padx=5, pady=50)
        low_level = self.ini.get_option("spectrograph", "low_level")
        self.entry_low.insert(0, low_level)

        label_sample = tk.Label(parameter_window, text=f"采样时间")
        label_sample.grid(row=3, column=0, padx=5, pady=5)

        self.entry_sample = tk.Entry(parameter_window)
        self.entry_sample.grid(row=3, column=1, padx=5, pady=5)
        sample_protect = self.ini.get_option("sample", "closing_time")
        self.entry_sample.insert(0, sample_protect)

        label_drain = tk.Label(parameter_window, text=f"消解时间")
        label_drain.grid(row=3, column=2, padx=5, pady=5)

        self.entry_drain = tk.Entry(parameter_window)
        self.entry_drain.grid(row=3, column=3, padx=5, pady=5)
        drain_process = self.ini.get_option("drain_process", "stop")
        self.entry_drain.insert(0, drain_process)

        # 添加提交按钮
        # submit_button = tk.Button(parameter_window, text="提交", command=self.on_submit)
        # submit_button.grid(row=20, column=0, columnspan=2, padx=5, pady=5)

        # submit_button = tk.Button(parameter_window, text="保存", command=self.on_params_save)  #  暂时屏蔽
        # submit_button.grid(row=15, column=3, ipadx=10, padx=10, pady=50)
        logger.info("Parameter Control clicked")

    def open_timer_control(self):
        # 这里可以添加打开参数控制界面的代码

        # 这里可以添加打开控制单元界面的代码
        # 创建新的Toplevel窗口
        timer_window = tk.Toplevel(self)
        # 使窗口变为模态
        # timer_window.transient(self)
        # timer_window.grab_set()
        # button1.grid(row=10, column=5, padx=5, pady=5)

        # 设置窗口标题和大小
        timer_window.title("定时")
        # parameter_window.geometry("600x400")
        center_window(timer_window, 720, 500)

        # 创建 IntVar 变量来存储复选框的状态
        self.init_hours = [tk.IntVar() for _ in range(24)]
        self.service_hours = [tk.IntVar() for _ in range(24)]

        init_values = self.ini.get_str_option("timer", "init")
        if init_values:
            for index in init_values.split(","):
                self.init_hours[int(index)].set(1)

        service_values = self.ini.get_str_option("timer", "service")
        if service_values:
            for index in service_values.split(","):
                self.service_hours[int(index)].set(1)
        # 创建 Checkbutton 控件并添加到窗口中
        label1 = tk.Label(timer_window, text="初始化")
        label1.grid(row=1, column=1, padx=3, pady=20)
        for i in range(24):
            check_button_init_hour = tk.Checkbutton(
                timer_window, text=f"{i}点", variable=self.init_hours[i]
            )
            check_button_init_hour.grid(
                row=int((i / 12) + 1 + 1), column=(i % 12) + 1, padx=3, pady=20
            )

        label2 = tk.Label(timer_window, text="检测流程")
        label2.grid(row=4, column=1, padx=3, pady=20)
        for i in range(24):
            check_button_service_hour = tk.Checkbutton(
                timer_window, text=f"{i}点", variable=self.service_hours[i]
            )
            check_button_service_hour.grid(
                row=int((i / 12) + 4 + 1), column=(i % 12) + 1, padx=3, pady=20
            )
        # 添加提交按钮
        # submit_button = tk.Button(parameter_window, text="提交", command=self.on_submit)
        # submit_button.grid(row=20, column=0, columnspan=2, padx=5, pady=5)
        submit_button = tk.Button(timer_window, text="保存", command=self.on_timer_save)
        submit_button.grid(row=15, columnspan=2, column=7, ipadx=10, padx=10, pady=10)
        logger.info("Timer Control clicked")

    def on_Y_click(self, slave, index):
        if slave == 1:
            logger.info("Y{}复选框的值= {}", index + 1, self.Y_check_vars[index].get())
            if self.Y_check_vars[index].get() == 1:
                self.Y_check_vars[index].set(1)
                self.ctrl.set_single_coil(slave, index, 1)
                logger.info("Y{}状态已设置为闭合", index + 1)
            else:
                self.Y_check_vars[index].set(0)
                self.ctrl.set_single_coil(slave, index, 0)
                logger.info("Y{}状态已设置为断开", index + 1)
        elif slave == 2:
            logger.info(
                "1_Y{}复选框的值= {}", index + 1, self.Y_1_check_vars[index].get()
            )
            if self.Y_1_check_vars[index].get() == 1:
                self.Y_1_check_vars[index].set(1)
                self.ctrl.set_single_coil(slave, index, 1)
                logger.info("1_Y{}状态已设置为闭合", index + 1)
            else:
                self.Y_1_check_vars[index].set(0)
                self.ctrl.set_single_coil(slave, index, 0)
                logger.info("1_Y{}状态已设置为断开", index + 1)
        IO1_all_coils, IO2_all_coils = self.ctrl.get_all_coils()
        logger.info(
            "Y_click操作后IO1_all_coils={} IO2_all_coils={}",
            IO1_all_coils,
            IO2_all_coils,
        )
        self.load_all_X()

    def on_Y_all_click(self, slave):
        for i in range(16):
            if slave == 1:
                self.Y_check_vars[i].set(0)
            elif slave == 2:
                self.Y_1_check_vars[i].set(0)
            self.ctrl.set_single_coil(slave, i, 0)
        IO1_all_coils, IO2_all_coils = self.ctrl.get_all_coils()
        logger.info(
            "Y_all_click 操作后IO1_all_coils={} IO2_all_coils={}",
            IO1_all_coils,
            IO2_all_coils,
        )
        self.load_all_X()

    def open_device_control(self):
        # 这里可以添加打开设备控制界面的代码

        # 创建新的Toplevel窗口
        device_window = tk.Toplevel(self)
        # 使窗口变为模态
        # device_window.transient(self)
        # device_window.grab_set()
        # button1.grid(row=10, column=5, padx=5, pady=5)

        # 设置窗口标题和大小
        device_window.title("设备控制")
        # parameter_window.geometry("600x400")
        center_window(device_window, 1050, 500)

        # 创建 IntVar 变量来存储复选框的状态
        self.X_check_vars = [tk.IntVar() for _ in range(16)]
        self.X_1_check_vars = [tk.IntVar() for _ in range(16)]
        self.Y_check_vars = [tk.IntVar() for _ in range(16)]
        self.Y_1_check_vars = [tk.IntVar() for _ in range(16)]

        # 创建 Checkbutton 控件并添加到窗口中
        for i in range(16):
            check_button_X = tk.Checkbutton(
                device_window, text=f"X{i + 1}", variable=self.X_check_vars[i]
            )
            check_button_X.configure(state="disabled")
            check_button_X.grid(row=3, column=i % 16, padx=3, pady=20)
            check_button_X1 = tk.Checkbutton(
                device_window, text=f"1_X{i + 1}", variable=self.X_1_check_vars[i]
            )
            check_button_X1.configure(state="disabled")
            check_button_X1.grid(row=4, column=i % 16, padx=3, pady=20)

            check_button_Y = tk.Checkbutton(
                device_window,
                text=f"Y{i + 1}",
                variable=self.Y_check_vars[i],
                command=lambda index=i: self.on_Y_click(1, index),
            )
            check_button_Y.grid(row=5, column=i % 16, padx=3, pady=20)
            check_button_Y1 = tk.Checkbutton(
                device_window,
                text=f"1_Y{i + 1}",
                variable=self.Y_1_check_vars[i],
                command=lambda index=i: self.on_Y_click(2, index),
            )
            check_button_Y1.grid(row=6, column=i % 16, padx=3, pady=20)

        close_button = tk.Button(
            device_window, text="Y 全部断开", command=lambda: self.on_Y_all_click(1)
        )
        close_button.grid(row=30, column=6, columnspan=3, ipadx=50, pady=20)

        close_button = tk.Button(
            device_window, text="1_Y 全部断开", command=lambda: self.on_Y_all_click(2)
        )
        close_button.grid(row=30, column=9, columnspan=3, ipadx=50, pady=20)

        self.load_all_X()

        IO1_all_coils, IO2_all_coils = self.ctrl.get_all_coils()
        for i in range(16):
            self.Y_check_vars[i].set(IO1_all_coils[i])
            self.Y_1_check_vars[i].set(IO2_all_coils[i])
        # logger.info("y_check_vars={} y_1_check_vars={}", self.Y_check_vars, self.Y_1_check_vars)

        logger.info("Device Control clicked")

    def load_all_X(self):

        IO1_all_input, IO2_all_input = self.ctrl.get_all_input()
        for i in range(16):
            self.X_check_vars[i].set(IO1_all_input[i])
            self.X_1_check_vars[i].set(IO2_all_input[i])
        logger.info(
            "X_check_vars={} X_1_check_vars={}", self.X_check_vars, self.X_1_check_vars
        )

    def on_timer_save(self):
        logger.info(f"定时任务保存 clicked")
        init_value = ",".join(
            str(index) for index, hour in enumerate(self.init_hours) if hour.get() == 1
        )
        self.ini.set_option("timer", "init", init_value)

        service_value = ",".join(
            str(index)
            for index, hour in enumerate(self.service_hours)
            if hour.get() == 1
        )
        self.ini.set_option("timer", "service", service_value)
        self.task.remove_jobs("init")
        self.task.remove_jobs("service")
        self.task.add_init_job()
        self.task.add_service_job()
        logger.info(f"初始化 全流程定时任务已更新")

    def on_params_save(self):
        self.ini.set_option("spectrograph", "score", self.entry_score.get())
        self.ini.set_option("spectrograph", "average_time", self.entry_avg.get())
        self.ini.set_option("spectrograph", "high_level", self.entry_high.get())
        self.ini.set_option("spectrograph", "low_level", self.entry_low.get())
        self.ini.set_option("sample", "closing_time", self.entry_sample.get())
        self.ini.set_option("drain_process", "stop", self.entry_drain.get())
        logger.info(f"参数保存 clicked")

    def open_control_unit(self):
        # 这里可以添加打开控制单元界面的代码
        # 创建新的Toplevel窗口
        control_window = tk.Toplevel(self)
        # 使窗口变为模态
        # control_window.transient(self)
        # control_window.grab_set()
        # 创建按钮
        button_frame = tk.Frame(control_window)
        button_frame.pack(pady=10)

        button1 = tk.Button(button_frame, text=f"采样回路", command=self.sample).grid(
            row=0, column=0, padx=5, pady=10
        )
        button2 = tk.Button(button_frame, text=f"排空回路", command=self.empty).grid(
            row=0, column=1, padx=5, pady=10
        )
        button3 = tk.Button(
            button_frame, text=f"加样-原⽔-检测池", command=self.sample_water_pool
        ).grid(row=0, column=2, padx=5, pady=10)
        button4 = tk.Button(
            button_frame, text=f"排空-检测-原⽔池", command=self.empty_check_water
        ).grid(row=0, column=3, padx=5, pady=10)
        button5 = tk.Button(
            button_frame, text=f"加样-检测-消解池", command=self.water_ctrl_water
        ).grid(row=0, column=4, padx=5, pady=10)

        button6 = tk.Button(
            button_frame, text=f"排空消解-检测池", command=self.water_empty_water
        ).grid(row=1, column=0, padx=5, pady=10)
        button7 = tk.Button(
            button_frame, text=f"光程切换10MM", command=self.pool_light_switch
        ).grid(row=1, column=1, padx=5, pady=10)
        button8 = tk.Button(
            button_frame, text=f"电解-电解", command=self.elctr_elctr
        ).grid(row=1, column=2, padx=5, pady=10)
        button9 = tk.Button(
            button_frame, text=f"消解加臭氧", command=self.drain_O3
        ).grid(row=1, column=3, padx=5, pady=10)
        button10 = tk.Button(
            button_frame, text=f"检测池加臭氧", command=self.check_O3
        ).grid(row=1, column=4, padx=5, pady=10)

        button11 = tk.Button(
            button_frame, text=f"检测池加纯⽔", command=self.check_water
        ).grid(row=2, column=0, padx=5, pady=10)
        button11_1 = tk.Button(
            button_frame, text=f"电解池加纯⽔", command=self.elctr_water
        ).grid(row=2, column=1, padx=5, pady=10)
        button11_2 = tk.Button(
            button_frame, text=f"原水池加纯⽔", command=self.water_water
        ).grid(row=2, column=2, padx=5, pady=10)
        button11_3 = tk.Button(
            button_frame, text=f"系统自检", command=self.self_test
        ).grid(row=2, column=3, padx=5, pady=10)

        button12 = tk.Button(
            button_frame, text=f"消解-超声控制", command=self.drain_sonic
        ).grid(row=3, column=0, padx=5, pady=10)
        button13 = tk.Button(
            button_frame, text=f"消解-UV控制", command=self.drain_uv
        ).grid(row=3, column=1, padx=5, pady=10)
        button14 = tk.Button(
            button_frame, text=f"原⽔-超声控制", command=self.water_sonic
        ).grid(row=3, column=2, padx=5, pady=10)
        button15 = tk.Button(
            button_frame, text=f"检测-超声控制", command=self.detect_sonic
        ).grid(row=3, column=3, padx=5, pady=10)

        button16 = tk.Button(
            button_frame, text=f"检测模块-⽔质传感器", command=self.detect_water_sensor
        ).grid(row=4, column=0, padx=5, pady=10)
        button17 = tk.Button(
            button_frame, text=f"原⽔模块-⽔质传感器", command=self.water_water_sensor()
        ).grid(row=4, column=1, padx=5, pady=10)
        button18 = tk.Button(button_frame, text=f"温控-机箱温度", command=None).grid(
            row=4, column=2, padx=5, pady=10
        )
        button19 = tk.Button(
            button_frame, text=f"电池电量读数", command=self.get_battery
        ).grid(row=4, column=3, padx=5, pady=10)
        button20 = tk.Button(button_frame, text=f"信号⽔平读数", command=None).grid(
            row=4, column=4, padx=5, pady=10
        )

        button21 = tk.Button(button_frame, text=f"报警信息", command=None).grid(
            row=5, column=0, padx=5, pady=10
        )
        button22 = tk.Button(button_frame, text=f"信号弱报警", command=None).grid(
            row=5, column=1, padx=5, pady=10
        )
        button23 = tk.Button(button_frame, text=f"电解模块报警", command=None).grid(
            row=5, column=2, padx=5, pady=10
        )
        button24 = tk.Button(button_frame, text=f"电解模块⽓压报警", command=None).grid(
            row=5, column=3, padx=5, pady=10
        )
        button25 = tk.Button(
            button_frame, text=f"光程切换30MM", command=self.pool_light_switch30
        ).grid(row=5, column=4, padx=5, pady=10)

        button26 = tk.Button(
            button_frame, text=f"消解加臭氧 UV 超声", command=self.drain_O3_uv_sonic
        ).grid(row=6, column=0, padx=5, pady=10)

        button26_1 = tk.Button(
            button_frame, text=f"消解流程", command=self.drain_process
        ).grid(row=6, column=1, padx=5, pady=10)

        button26_2 = tk.Button(
            button_frame, text=f"消解去臭氧", command=self.drain_remove_O3
        ).grid(row=6, column=2, padx=5, pady=10)
        button26_3 = tk.Button(
            button_frame, text=f"读取COD", command=self.read_cod
        ).grid(row=6, column=3, padx=5, pady=10)

        button26_4 = tk.Button(
            button_frame, text=f"最新预测数据", command=self.lastest_data
        ).grid(row=6, column=4, padx=5, pady=10)

        button27 = tk.Button(
            button_frame, text=f"测量背景数据", command=self.measure_background_data
        ).grid(row=7, column=0, padx=5, pady=10)
        button28 = tk.Button(
            button_frame, text=f"测量标准数据", command=self.measure_standand_data
        ).grid(row=7, column=1, padx=5, pady=10)
        button29 = tk.Button(
            button_frame, text=f"测量样品数据", command=self.measure_sample_data
        ).grid(row=7, column=2, padx=5, pady=10)
        button30 = tk.Button(
            button_frame, text=f"计算吸光度", command=self.calc_absorbance
        ).grid(row=7, column=3, padx=5, pady=10)
        button32 = tk.Button(
            button_frame, text=f"上传吸光度", command=self.calc_absorbance
        ).grid(row=7, column=4, padx=5, pady=10)

        # button33 = tk.Button(button_frame, text=f"标样全自动流程",
        #                      command=self.measure_background_data).grid(row=8, column=0, padx=5, pady=10)
        action = "PREDICT_STANDARD_TP"
        button33 = tk.Button(
            button_frame,
            text="标样全自动总磷",
            command=self.measure_standand_service_tp,
        ).grid(row=8, column=0, padx=5, pady=10)
        button34 = tk.Button(
            button_frame,
            text="标样全自动总氮",
            command=self.measure_standand_service_tn,
        ).grid(row=8, column=1, padx=5, pady=10)
        button35 = tk.Button(
            button_frame,
            text="标样全自动COD",
            command=self.measure_standand_service_cod,
        ).grid(row=8, column=2, padx=5, pady=10)
        button36 = tk.Button(
            button_frame,
            text="标样全自动氨氮",
            command=self.measure_standand_service_an,
        ).grid(row=8, column=3, padx=5, pady=10)
        button37 = tk.Button(
            button_frame,
            text="标样全自动高锰酸钾",
            command=self.measure_standand_service_kmno,
        ).grid(row=8, column=4, padx=5, pady=10)
        button38 = tk.Button(
            button_frame,
            text="标样全自动浊度",
            command=self.measure_standand_service_tur,
        ).grid(row=8, column=5, padx=5, pady=10)
        button39 = tk.Button(
            button_frame,
            text="标样全自动零标",
            command=self.measure_standand_service_zero,
        ).grid(row=8, column=6, padx=5, pady=10)

        button33 = tk.Button(
            button_frame, text="总磷标样", command=self.measure_tp_standand_service
        ).grid(row=9, column=0, padx=5, pady=10)
        button34 = tk.Button(
            button_frame, text="总氮标样", command=self.measure_tn_standand_service
        ).grid(row=9, column=1, padx=5, pady=10)
        button35 = tk.Button(
            button_frame, text="COD标样", command=self.measure_cod_standand_service
        ).grid(row=9, column=2, padx=5, pady=10)
        button36 = tk.Button(
            button_frame, text="氨氮标样", command=self.measure_an_standand_service
        ).grid(row=9, column=3, padx=5, pady=10)
        button37 = tk.Button(
            button_frame,
            text="高锰酸盐标样",
            command=self.measure_kmno_standand_service,
        ).grid(row=9, column=4, padx=5, pady=10)
        button38 = tk.Button(
            button_frame, text="浊度标样", command=self.measure_tur_standand_service
        ).grid(row=9, column=5, padx=5, pady=10)

        button39 = tk.Button(
            button_frame, text=f"盲样全自动流程", command=self.measure_blind_service
        ).grid(row=9, column=6, padx=5, pady=10)

        button_init = tk.Button(
            button_frame, text=f"初始化", command=self.on_initialize
        ).grid(row=10, column=1, padx=5, pady=10)
        button_init2 = tk.Button(
            button_frame, text=f"循环初始化", command=self.on_initialize2
        ).grid(row=10, column=2, padx=5, pady=10)
        button_fullservice = tk.Button(
            button_frame, text=f"自动检测全流程", command=self.on_check
        ).grid(row=10, column=3, padx=5, pady=10)
        button_fullservice_one = tk.Button(
            button_frame, text=f"自动检测一次", command=self.on_check_one
        ).grid(row=10, column=4, padx=10, pady=10)
        button_fullservice_one = tk.Button(
            button_frame, text=f"消解配方", command=self.on_digest
        ).grid(row=10, column=5, padx=10, pady=10)
        # 设置窗口标题和大小
        control_window.title("控制单元")
        # control_window.geometry("700x500")
        center_window(control_window, 840, 600)

        logger.info("control Unit clicked")

    @logger.catch()
    def sample(self):
        t = threading.Thread(target=self.ctrl.sample)
        t.start()

    def empty(self):
        t = threading.Thread(target=self.ctrl.empty)
        t.start()

    def sample_water_pool(self):
        t = threading.Thread(target=self.ctrl.sample_water_pool)
        t.start()

    def empty_check_water(self):
        t = threading.Thread(target=self.ctrl.empty_check_water)
        t.start()

    def water_ctrl_water(self):
        t = threading.Thread(target=self.ctrl.water_ctrl_water)
        t.start()

    def water_empty_water(self):
        t = threading.Thread(target=self.ctrl.water_empty_water)
        t.start()

    def pool_light_switch(self):
        t = threading.Thread(target=self.ctrl.pool_light_switch10)
        t.start()

    def elctr_elctr(self):
        t = threading.Thread(target=self.ctrl.elctr_elctr)
        t.start()

    def pool_light_switch30(self):
        t = threading.Thread(target=self.ctrl.pool_light_switch30)
        t.start()

    def drain_O3_uv_sonic(self):
        t1 = threading.Thread(target=self.ctrl.drain_O3)
        t2 = threading.Thread(target=self.ctrl.drain_uv)
        t3 = threading.Thread(target=self.ctrl.drain_sonic)

        # 启动线程
        t1.start()
        t2.start()
        t3.start()

        """消解流程"""

    def drain_process(self):
        t1 = threading.Thread(target=self.ctrl.drain_process)
        t3 = threading.Thread(target=self.ctrl.drain_sonic_pulse)
        # 启动线程
        t1.start()
        t3.start()

    """消解去臭氧"""

    def drain_remove_O3(self):
        t1 = threading.Thread(target=self.ctrl.drain_remove_O3)
        t1.start()

    def read_cod(self):
        cs = CODSensor()
        # cs.clear_sensor()
        cod_value, temp_value, turbidity_value = cs.read_cod_data()
        cs.close()
        messagebox.showinfo(
            "提示",
            "cod:"
            + str(cod_value)
            + " temp:"
            + str(temp_value)
            + " turbidity:"
            + str(turbidity_value),
        )

    def lastest_data(self):
        data = self.ctrl.get_lastest_data()
        dict_str = json.dumps(data["data"])
        messagebox.showinfo("提示", dict_str)

    def measure_background_data(self):
        logger.info("测量背景数据 clicked")
        NO = util.get_background_NO()
        self.spectrograph.params_setup()
        t1 = threading.Thread(
            target=self.spectrograph.measure_report_10_background_data,
            args=(NO, TypeEnum.ORIGIN.value, OpticalLengthEnum.MM_10.value),
        )  # measure_background_data
        # 启动线程
        t1.start()

    def measure_standand_data(self):
        logger.info("测量纯水数据 clicked")
        NO = util.get_standard_NO()
        self.spectrograph.params_setup()
        t1 = threading.Thread(
            target=self.spectrograph.measure_report_standard_data,
            args=(NO, TypeEnum.ORIGIN.value, OpticalLengthEnum.MM_10.value),
        )  # measure_standard_data
        # 启动线程
        t1.start()

    def measure_standand_service_tp(self):
        logger.info("测量标准数据流程总磷  clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.autoDetect, args=("PREDICT_STANDARD_TP",)
        )
        # 启动线程
        t1.start()

    def measure_standand_service_tn(self):
        logger.info("测量标准数据流程总氮 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.autoDetect, args=("PREDICT_STANDARD_TN",)
        )
        # 启动线程
        t1.start()

    def measure_standand_service_cod(self):
        logger.info("测量标准数据流程cod clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.autoDetect, args=("PREDICT_STANDARD_COD",)
        )
        # 启动线程
        t1.start()

    def measure_standand_service_an(self):
        logger.info("测量标准数据流程氨氮 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.autoDetect, args=("PREDICT_STANDARD_AN",)
        )
        # 启动线程
        t1.start()

    def measure_standand_service_kmno(self):
        logger.info("测量标准数据流程高锰酸钾 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.autoDetect, args=("PREDICT_STANDARD_KMNO",)
        )
        # 启动线程
        t1.start()

    def measure_standand_service_tur(self):
        logger.info("测量标准数据流程浊度 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.autoDetect, args=("PREDICT_STANDARD_TUR",)
        )
        # 启动线程
        t1.start()

    def measure_standand_service_zero(self):
        logger.info("测量标准数据流程零标 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.autoDetect, args=("PREDICT_STANDARD_ZERO",)
        )
        # 启动线程
        t1.start()

    def measure_tp_standand_service(self):
        logger.info("测量总磷标液流程  clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.spectrumDetect, args=("PREDICT_STANDARD_TP",)
        )
        # 启动线程
        t1.start()

    def measure_tn_standand_service(self):
        logger.info("测量总氮标液流程 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.spectrumDetect, args=("PREDICT_STANDARD_TN",)
        )
        # 启动线程
        t1.start()

    def measure_cod_standand_service(self):
        logger.info("测量cod标液流程 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.spectrumDetect, args=("PREDICT_STANDARD_COD",)
        )
        # 启动线程
        t1.start()

    def measure_an_standand_service(self):
        logger.info("测量氨氮标液流程 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.spectrumDetect, args=("PREDICT_STANDARD_AN",)
        )
        # 启动线程
        t1.start()

    def measure_kmno_standand_service(self):
        logger.info("测量高锰酸盐标液流程 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.spectrumDetect, args=("PREDICT_STANDARD_KMNO",)
        )
        # 启动线程
        t1.start()

    def measure_tur_standand_service(self):
        logger.info("测量浊度标液流程 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(
            target=full_service_standard.spectrumDetect, args=("PREDICT_STANDARD_TUR",)
        )
        # 启动线程
        t1.start()

    def measure_blind_service(self):
        logger.info("盲样全自动流程 clicked ")
        NO = util.get_standard_NO()
        t1 = threading.Thread(target=full_service_blind.autoDetect)
        # 启动线程
        t1.start()

    def measure_sample_data(self):
        logger.info("测量采样数据 clicked")
        NO = util.get_sample_NO()
        self.spectrograph.params_setup()
        t1 = threading.Thread(
            target=self.spectrograph.measure_sample_data,
            args=(
                NO,
                ActionEnum.ADD_GLT_SAMPLE.value,
                TypeEnum.UV.value,
                OpticalLengthEnum.MM_10.value,
            ),
        )
        # 启动线程
        t1.start()

    def calc_absorbance(self):
        logger.info("计算吸光度 clicked")
        t1 = threading.Thread(target=self.spectrograph.calc_absorbance)
        # 启动线程
        t1.start()

    def drain_O3(self):
        t = threading.Thread(target=self.drain_O3)
        t.start()

    def check_O3(self):
        t = threading.Thread(target=self.ctrl.check_O3)
        t.start()

    def check_water(self):
        t = threading.Thread(target=self.ctrl.check_water)
        t.start()

    def elctr_water(self):
        t = threading.Thread(target=self.ctrl.elctr_water)
        t.start()

    def water_water(self):
        t = threading.Thread(target=self.ctrl.water_water)
        t.start()

    def self_test(self):
        t = threading.Thread(target=self.ctrl.self_test)
        t.start()

    def drain_sonic(self):
        t = threading.Thread(target=self.ctrl.drain_sonic)
        t.start()

    def drain_uv(self):
        t = threading.Thread(target=self.ctrl.drain_uv)
        t.start()

    def water_sonic(self):
        t = threading.Thread(target=self.ctrl.water_sonic)
        t.start()

    def detect_sonic(self):
        t = threading.Thread(target=self.ctrl.detect_sonic)
        t.start()

    """ 电池电量"""

    def get_battery(self):
        amount_value = self.ctrl.get_battery_router()
        messagebox.showinfo("提示", "电池电量为" + str(amount_value) + "%")

    """检测 水质传感器"""

    def detect_water_sensor(self):
        ph = PHSensor()
        ph_value, temp_value = ph.read_PH_data()
        ph.close()
        messagebox.showinfo(
            "提示", "PH为" + str(ph_value) + "\n温度为" + str(temp_value)
        )

    """原水 水质传感器"""

    def water_water_sensor(self):
        pass

    def test_drift(self, action, title, type, isTur=False):
        self.show_input_dialog(action, title, type, isTur)

    def test_lowerlimit(self, action, title, type, isTur=False):
        self.show_input_dialog(action, title, type, isTur)

    def test_indicationerror(self, action, title, type, isTur=False):
        self.show_input_dialog(action, title, type, isTur)

    def test_repeatability(self, action, title, type, isTur=False):
        self.show_input_dialog(action, title, type, isTur)

    def test_consistency(self, action, title, type, isTur=False):
        self.show_input_dialog(action, title, type, isTur)

    def show_input_dialog(self, action, test_title, type, isTur=False):
        # 创建模态对话框
        title = test_title
        if "COD" == type:
            title = "化学需氧量：" + title
        elif "TP" == type:
            title = "总磷：" + title
        elif "KMNO" == type:
            title = "高锰酸盐：" + title
        elif "AN" == type:
            title = "氨氮：" + title
        elif "TN" == type:
            title = "总氮：" + title
        elif "TURBIDITY" in type:
            title = "浊度：" + title

        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("300x200")
        dialog.resizable(False, False)

        # 居中显示窗口
        window_width = 300
        window_height = 200
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        dialog.geometry(
            f"{window_width}x{window_height}+{position_right}+{position_top}"
        )

        # 设置标签
        label = tk.Label(dialog, text="选择测试类型：")
        label.pack(pady=(20, 5))

        # 创建下拉框并填充枚举值
        if isTur:
            exam_types = list(TurExamType)
        else:
            exam_types = list(ExamType)

        exam_types = list(filter(lambda x: test_title in x.value, exam_types))
        selected_value = tk.StringVar(dialog)
        selected_value.set(exam_types[0].value)  # 默认选择第一个枚举项
        dropdown = ttk.Combobox(
            dialog,
            textvariable=selected_value,
            values=[et.value for et in exam_types],
            state="readonly",
        )
        dropdown.pack(pady=5)

        # 定义按钮事件
        def on_test():
            # 获取所选值和索引
            selected_index = dropdown.current()
            selected_text = selected_value.get()
            print(f"所选的下拉框索引：{selected_index}")
            print(f"所选的下拉框值：{selected_text}")
            even_impl(action, type, 1, selected_index)
            dialog.destroy()  # 关闭对话框

        def on_close():
            dialog.destroy()  # 直接关闭对话框

        def even_impl(action, type, detect_count, selected_index):
            t = threading.Thread(
                target=full_service_standard.spectrumDetectMore,
                args=(action, detect_count, type, selected_index),
            )
            t.start()

        # 创建按钮容器（用于水平布局按钮）
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)

        # 创建测试按钮和关闭按钮
        test_button = tk.Button(button_frame, text="测试", command=on_test)
        test_button.pack(side=tk.LEFT, padx=10)

        close_button = tk.Button(button_frame, text="关闭", command=on_close)
        close_button.pack(side=tk.LEFT, padx=10)

        # 设置模态窗口
        dialog.transient(self)
        dialog.grab_set()  # 将焦点设置到对话框上
        self.wait_window(dialog)  # 等待对话框关闭后再继续执行

    def open_about(self):
        # 这里可以添加打开控制单元界面的代码
        # 创建新的Toplevel窗口
        about_window = tk.Toplevel(self)
        # 使窗口变为模态
        # about_window.transient(self)
        # about_window.grab_set()
        # 创建按钮
        button_frame = tk.Frame(about_window)
        button_frame.pack(pady=10)
        button1 = tk.Button(button_frame, text=f"关闭", command=about_window.destroy)
        button1.pack(side=tk.BOTTOM, padx=(100, 100))
        # 设置窗口标题和大小
        about_window.title("关于")
        # about_window.geometry("300x200")
        center_window(about_window, 300, 200)
        logger.info("about_window clicked")

    def on_close(self):
        logger.info("Closing app...")
        try:
            signal_client.report_close()
            self.ctrl.reset_ctrl(True)
            logger.info("窗口已经关闭")
        except Exception as e:
            logger.error(e)
        finally:
            app.destroy()

    def on_initialize(self):
        logger.info("初始化Initialize clicked")
        t = threading.Thread(target=init.run)
        t.start()

    def on_initialize2(self):
        logger.info("循环初始化Initialize clicked")
        t = threading.Thread(target=init.run2)
        t.start()

    def is_time_in_range(self, start_time, end_time, current_time):
        """检查当前时间是否在给定的时间范围内"""
        return start_time <= current_time <= end_time

    def method_to_run(self):
        full_service.spectrograph_setup()
        while not stop_event.is_set():
            logger.info("method_to_run方法正在运行...")
            # service.autoDetect()
            try:
                full_service.autoDetect(self)

                # current_time = datetime.datetime.now().time()
                # start_time =  datetime.time(0, 0, 0)  # 0:00
                # end_time =  datetime.time(0, 30, 0)  # 0:30
                # if self.is_time_in_range(start_time, end_time, current_time):
                #     logger.info("自动检测流程初始化开始===========（1）")
                #     init.spectrograph_init()
                #     logger.info("自动检测流程初始化结束===========（2）")
            except Exception as e:
                logger.info("自动检测运行异常===========X")
                logger.error(e)
            finally:
                self.ctrl.reset_ctrl()
        logger.info("method_to_run方法已停止")

    def digest_to_run(self):
        full_service.spectrograph_setup()
        count = 0
        while not digest_event.is_set():
            logger.info("digest_to_run方法正在运行...")
            try:
                if count >= 7:
                    break
                full_service.digestDetect(self)
                count += 1
            except Exception as e:
                logger.info("消解配方异常===========X")
                logger.error(e)
            finally:
                self.ctrl.reset_ctrl()
        logger.info("digest_to_run方法已停止")

    def method_to_run_one(self):
        full_service.autoDetect(self)
        logger.info("method_to_run_one方法已停止")

    def on_check(self):
        global stop_event
        if stop_event.is_set():
            stop_event.clear()
            t = threading.Thread(target=self.method_to_run)
            t.start()
            logger.info("自动检测运行 开始==========")
        else:
            stop_event.set()
            logger.info("自动检测运行 停止===========")
        logger.info("自动检测运行 clicked")

    def on_check_one(self):
        t = threading.Thread(target=self.method_to_run_one)
        t.start()
        logger.info("自动检测一次运行 clicked==========>")

    def on_digest(self):
        global digest_event
        if digest_event.is_set():
            digest_event.clear()
            t = threading.Thread(target=self.digest_to_run)
            t.start()
            logger.info("消解配方流程运行 开始==========")
        else:
            digest_event.set()
            logger.info("消解配方流程运行 停止===========")
        logger.info("消解配方流程 clicked")

    def on_close_ctrl(self):
        logger.info("重置按钮 clicked")
        t = threading.Thread(target=self.ctrl.reset_ctrl)
        t.start()


def center_window(window, width, height):
    # 获取屏幕尺寸
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # 计算窗口的宽度和高度
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    # 设置窗口的位置
    window.geometry(f"{width}x{height}+{x}+{y}")


# def set_background(root, image_path):
#     logger.info("设置背景{}",image_path)
#     # Load the image
#     img = Image.open(image_path)
#     # Resize the image to fit the window size
#     img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.ANTIALIAS)
#     # Create a PhotoImage object from the image
#     photo = ImageTk.PhotoImage(img)
#     # Create a label to hold the image
#     label = tk.Label(root, image=photo)
#     label.image = photo  # This line is necessary to prevent the image from being garbage collected
#     label.place(x=0, y=0, relwidth=1, relheight=1)
if __name__ == "__main__":
    app = MainFrame()

    # 系统重置
    t = threading.Thread(target=app.ctrl.reset_ctrl)
    t.start()
    # 在界面启动时调用my_method方法
    # app.after(0, app.on_initialize())
    # app.after(0, app.ctrl.reset_ctrl)
    # set_background(app,'1920.jpg')
    app.mainloop()

    # init.run()
    # stop_event.clear()
    # app.on_check()
