import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import re
from datetime import datetime
import os
import json
import threading
import sqlite3
from typing import Tuple, List, Dict, Optional
import logging
from dataclasses import dataclass
import webbrowser
from pathlib import Path
import gc
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_parser.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


@dataclass
class AddressRecord:
    """地址记录数据类"""
    name: str = ""
    phone: str = ""
    address: str = ""
    created_time: str = ""
    updated_time: str = ""
    tags: str = ""
    notes: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'phone': self.phone,
            'address': self.address,
            'created_time': self.created_time,
            'updated_time': self.updated_time,
            'tags': self.tags,
            'notes': self.notes,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AddressRecord':
        return cls(**data)


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: str = "address_parser.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS addresses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        phone TEXT,
                        address TEXT,
                        created_time TEXT,
                        updated_time TEXT,
                        tags TEXT,
                        notes TEXT,
                        confidence REAL DEFAULT 0.0
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS parsing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_text TEXT,
                        parsed_count INTEGER,
                        timestamp TEXT
                    )
                ''')

                # 创建索引提高查询性能
                conn.execute('CREATE INDEX IF NOT EXISTS idx_phone ON addresses(phone)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_name ON addresses(name)')

        except Exception as e:
            logging.error(f"Database initialization failed: {e}")

    def save_record(self, record: AddressRecord) -> bool:
        """保存记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO addresses (name, phone, address, created_time, updated_time, tags, notes, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (record.name, record.phone, record.address, record.created_time,
                      record.updated_time, record.tags, record.notes, record.confidence))
            return True
        except Exception as e:
            logging.error(f"Save record failed: {e}")
            return False

    def save_records_batch(self, records: List[AddressRecord]) -> bool:
        """批量保存记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = [(r.name, r.phone, r.address, r.created_time,
                         r.updated_time, r.tags, r.notes, r.confidence) for r in records]
                conn.executemany('''
                    INSERT INTO addresses (name, phone, address, created_time, updated_time, tags, notes, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', data)
            return True
        except Exception as e:
            logging.error(f"Batch save failed: {e}")
            return False

    def load_records(self, limit: int = 1000) -> List[AddressRecord]:
        """加载记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT name, phone, address, created_time, updated_time, tags, notes, confidence
                    FROM addresses ORDER BY updated_time DESC LIMIT ?
                ''', (limit,))

                records = []
                for row in cursor:
                    record = AddressRecord(
                        name=row[0] or "",
                        phone=row[1] or "",
                        address=row[2] or "",
                        created_time=row[3] or "",
                        updated_time=row[4] or "",
                        tags=row[5] or "",
                        notes=row[6] or "",
                        confidence=row[7] or 0.0
                    )
                    records.append(record)
                return records
        except Exception as e:
            logging.error(f"Load records failed: {e}")
            return []

    def search_records(self, keyword: str) -> List[AddressRecord]:
        """搜索记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT name, phone, address, created_time, updated_time, tags, notes, confidence
                    FROM addresses 
                    WHERE name LIKE ? OR phone LIKE ? OR address LIKE ? OR tags LIKE ?
                    ORDER BY updated_time DESC
                ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))

                records = []
                for row in cursor:
                    record = AddressRecord(
                        name=row[0] or "",
                        phone=row[1] or "",
                        address=row[2] or "",
                        created_time=row[3] or "",
                        updated_time=row[4] or "",
                        tags=row[5] or "",
                        notes=row[6] or "",
                        confidence=row[7] or 0.0
                    )
                    records.append(record)
                return records
        except Exception as e:
            logging.error(f"Search records failed: {e}")
            return []

    def save_parsing_history(self, original_text: str, parsed_count: int):
        """保存解析历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO parsing_history (original_text, parsed_count, timestamp)
                    VALUES (?, ?, ?)
                ''', (original_text[:500], parsed_count, datetime.now().isoformat()))
        except Exception as e:
            logging.error(f"Save parsing history failed: {e}")

    def get_parsing_history(self, limit: int = 50) -> List[Dict]:
        """获取解析历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT original_text, parsed_count, timestamp
                    FROM parsing_history ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
                return [{'text': row[0], 'count': row[1], 'time': row[2]} for row in cursor]
        except Exception as e:
            logging.error(f"Get parsing history failed: {e}")
            return []


class Config:
    """增强的配置管理类"""

    def __init__(self):
        self.config_file = "address_parser_config.json"
        self.default_config = {
            "last_save_path": str(Path.home() / "Desktop"),
            "window_geometry": "1400x900",
            "auto_parse": True,
            "export_format": "xlsx",
            "theme": "default",
            "auto_save": True,
            "backup_enabled": True,
            "max_history": 100,
            "font_size": 9,
            "preview_mode": "table",
            "show_tips": True,
            "language": "zh_CN"
        }
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return {**self.default_config, **json.load(f)}
        except Exception as e:
            logging.warning(f"Load config failed: {e}")
        return self.default_config.copy()

    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Save config failed: {e}")

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value):
        self.config[key] = value
        self.save_config()


class AdvancedAddressExtractor:
    """增强的地址信息提取器"""

    # 预编译正则表达式
    PHONE_PATTERNS = {
        'mobile': re.compile(
            r'(?<![0-9])((13[0-9]|14[01456879]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])\d{8})(?![0-9])'),
        'landline': re.compile(r'(?<![0-9])(\d{3,4}[-]?\d{7,8})(?![0-9])'),
        'international': re.compile(r'(\+86[-\s]?)?1[3-9]\d{9}')
    }

    ADDRESS_KEYWORDS = {
        'province': ['省', '自治区', '特别行政区'],
        'city': ['市', '地区', '盟', '州'],
        'district': ['区', '县', '旗'],
        'town': ['镇', '乡', '街道', '办事处'],
        'road': ['路', '街', '道', '大道', '小路', '巷', '弄'],
        'building': ['号', '栋', '楼', '层', '室', '门', '户'],
        'village': ['村', '庄', '屯', '组', '队']
    }

    NAME_PATTERNS = {
        'chinese': re.compile(r'^[\u4e00-\u9fa5]{2,4}$'),
        'english': re.compile(r'^[a-zA-Z\s]{2,20}$'),
        'mixed': re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9\s]{2,20}$')
    }

    @classmethod
    def extract_phone(cls, text: str) -> Tuple[str, str]:
        """提取电话号码，返回 (号码, 类型)"""
        # 优先匹配手机号
        for phone_type, pattern in cls.PHONE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                phone = match.group(1) if phone_type != 'international' else match.group(0)
                return phone.replace('-', '').replace(' ', ''), phone_type
        return "", ""

    @classmethod
    def extract_address(cls, text: str, exclude_parts: List[str] = None) -> Tuple[str, Dict]:
        """提取地址信息，返回 (地址, 地址组件)"""
        if exclude_parts:
            for part in exclude_parts:
                text = text.replace(part, ' ')

        # 清理文本
        text = re.sub(r'\s+', ' ', text).strip()

        # 分析地址组件
        address_components = {}
        for component, keywords in cls.ADDRESS_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    address_components[component] = True
                    break

        # 按逗号分割查找包含地址关键词的部分
        parts = [p.strip() for p in re.split(r'[,，]', text) if p.strip()]

        best_address = ""
        max_score = 0

        for part in parts:
            score = 0
            # 计算地址关键词数量
            for keywords in cls.ADDRESS_KEYWORDS.values():
                score += sum(1 for keyword in keywords if keyword in part)

            # 考虑长度因素
            if len(part) > 10:
                score += 1

            if score > max_score:
                best_address = part
                max_score = score

        return best_address, address_components

    @classmethod
    def extract_name(cls, text: str, exclude_parts: List[str] = None) -> Tuple[str, str]:
        """提取姓名，返回 (姓名, 类型)"""
        if exclude_parts:
            for part in exclude_parts:
                text = text.replace(part, ' ')

        # 清理文本
        text = re.sub(r'[,，]\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text or len(text) > 20:
            return "", ""

        # 检查是否包含地址关键词
        all_keywords = [kw for keywords in cls.ADDRESS_KEYWORDS.values() for kw in keywords]
        if any(keyword in text for keyword in all_keywords):
            return "", ""

        # 验证名字格式
        for name_type, pattern in cls.NAME_PATTERNS.items():
            if pattern.match(text):
                return text, name_type

        return "", ""

    @classmethod
    def extract_info(cls, text: str) -> Tuple[AddressRecord, Dict]:
        """提取完整信息，返回记录和元数据"""
        if not text or not text.strip():
            return AddressRecord(), {}

        text = text.strip()
        original_text = text

        # 统一标点和空白字符
        text = re.sub(r'[;；，]', ',', text)
        text = re.sub(r'\s+', ' ', text)

        # 提取信息
        phone, phone_type = cls.extract_phone(text)
        address, address_components = cls.extract_address(text, [phone] if phone else [])
        name, name_type = cls.extract_name(text, [phone, address] if phone and address else (
            [phone] if phone else [address] if address else []))

        # 创建记录
        now = datetime.now().isoformat()
        confidence = cls._calculate_confidence(name, phone, address)

        record = AddressRecord(
            name=name,
            phone=phone,
            address=address,
            created_time=now,
            updated_time=now,
            confidence=confidence
        )

        # 元数据
        metadata = {
            'original_text': original_text,
            'phone_type': phone_type,
            'name_type': name_type,
            'address_components': address_components,
            'confidence': confidence
        }

        return record, metadata

    @classmethod
    def _calculate_confidence(cls, name: str, phone: str, address: str) -> float:
        """计算置信度"""
        score = 0.0
        total = 3.0

        if name:
            score += 1.0
        if phone:
            score += 1.0
        if address:
            score += 1.0

        return score / total


class ModernAddressParser:
    """现代化地址解析器主类"""

    def __init__(self, root):
        self.root = root
        self.config = Config()
        self.db = DatabaseManager()
        self.setup_window()
        self.setup_variables()
        self.create_widgets()
        self.bind_events()
        self.load_window_state()

        # 自动保存定时器
        if self.config.get("auto_save", True):
            self.start_auto_save()

        logging.info("Application started successfully")

    def setup_window(self):
        """设置窗口属性"""
        self.root.title("AT打单专用 - 智能地址解析工具 v3.0")
        self.root.geometry(self.config.get("window_geometry", "1400x900"))
        self.root.minsize(1000, 700)

        # 设置主题
        self.setup_theme()

    def setup_theme(self):
        """设置主题"""
        style = ttk.Style()
        theme = self.config.get("theme", "default")

        try:
            if theme in style.theme_names():
                style.theme_use(theme)
        except:
            style.theme_use('default')

    def setup_variables(self):
        """设置变量"""
        self.auto_parse_var = tk.BooleanVar(value=self.config.get("auto_parse", True))
        self.auto_save_var = tk.BooleanVar(value=self.config.get("auto_save", True))
        self.status_var = tk.StringVar(value="就绪")
        self.search_var = tk.StringVar()
        self.filter_var = tk.StringVar(value="全部")

        # 数据相关
        self.current_records = []
        self.filtered_records = []
        self.selected_records = []

        # 统计信息
        self.stats = {
            'total_parsed': 0,
            'success_rate': 0.0,
            'last_parse_time': None
        }

    def create_widgets(self):
        """创建所有UI组件"""
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # 创建菜单栏
        self.create_menu()

        # 创建工具栏
        self.create_toolbar()

        # 创建主要内容区域
        self.create_paned_window()

        # 创建状态栏
        self.create_status_bar()

    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导入文本文件", command=self.import_text, accelerator="Ctrl+O")
        file_menu.add_command(label="导入Excel文件", command=self.import_excel)
        file_menu.add_separator()
        file_menu.add_command(label="导出Excel", command=self.export_excel, accelerator="Ctrl+E")
        file_menu.add_command(label="导出CSV", command=self.export_csv)
        file_menu.add_command(label="导出JSON", command=self.export_json)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing, accelerator="Ctrl+Q")

        # 编辑菜单
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="编辑", menu=edit_menu)
        edit_menu.add_command(label="解析文本", command=self.parse_text, accelerator="F5")
        edit_menu.add_command(label="清空输入", command=self.clear_input, accelerator="Ctrl+L")
        edit_menu.add_command(label="清空结果", command=self.clear_results)
        edit_menu.add_separator()
        edit_menu.add_command(label="全选", command=self.select_all, accelerator="Ctrl+A")
        edit_menu.add_command(label="反选", command=self.invert_selection)

        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="批量验证电话", command=self.validate_phones)
        tools_menu.add_command(label="地址标准化", command=self.normalize_addresses)
        tools_menu.add_command(label="去重", command=self.remove_duplicates)
        tools_menu.add_separator()
        tools_menu.add_command(label="统计分析", command=self.show_statistics)
        tools_menu.add_command(label="历史记录", command=self.show_history)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用指南", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)

    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.main_frame)
        toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # 主要操作按钮
        ttk.Button(toolbar, text="📂 导入", command=self.import_text).grid(row=0, column=0, padx=2)
        ttk.Button(toolbar, text="⚡ 解析", command=self.parse_text).grid(row=0, column=1, padx=2)
        ttk.Button(toolbar, text="💾 导出", command=self.export_excel).grid(row=0, column=2, padx=2)

        # 分隔符
        ttk.Separator(toolbar, orient='vertical').grid(row=0, column=3, sticky='ns', padx=5)

        # 编辑按钮
        ttk.Button(toolbar, text="✏️ 编辑", command=self.edit_selected).grid(row=0, column=4, padx=2)
        ttk.Button(toolbar, text="🗑️ 删除", command=self.delete_selected).grid(row=0, column=5, padx=2)
        ttk.Button(toolbar, text="🧹 清空", command=self.clear_all).grid(row=0, column=6, padx=2)

        # 分隔符
        ttk.Separator(toolbar, orient='vertical').grid(row=0, column=7, sticky='ns', padx=5)

        # 搜索框
        ttk.Label(toolbar, text="搜索:").grid(row=0, column=8, padx=(5, 2))
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=15)
        search_entry.grid(row=0, column=9, padx=2)
        search_entry.bind('<KeyRelease>', self.on_search_change)

        # 筛选下拉框
        ttk.Label(toolbar, text="筛选:").grid(row=0, column=10, padx=(10, 2))
        filter_combo = ttk.Combobox(toolbar, textvariable=self.filter_var, width=10, state='readonly')
        filter_combo['values'] = ('全部', '有姓名', '有电话', '有地址', '信息完整', '信息不全')
        filter_combo.grid(row=0, column=11, padx=2)
        filter_combo.bind('<<ComboboxSelected>>', self.on_filter_change)

        # 右侧选项
        ttk.Checkbutton(toolbar, text="自动解析", variable=self.auto_parse_var,
                        command=self.on_auto_parse_change).grid(row=0, column=12, padx=10)
        ttk.Checkbutton(toolbar, text="自动保存", variable=self.auto_save_var,
                        command=self.on_auto_save_change).grid(row=0, column=13, padx=5)

    def create_paned_window(self):
        """创建分割窗口"""
        # 主分割窗口（水平）
        main_paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        main_paned.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # 左侧面板
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=1)

        # 右侧面板
        right_frame = ttk.Frame(main_paned, width=800)
        main_paned.add(right_frame, weight=2)

        # 创建左侧内容
        self.create_input_panel(left_frame)

        # 创建右侧内容
        self.create_preview_panel(right_frame)

    def create_input_panel(self, parent):
        """创建输入面板"""
        # 输入区域
        input_frame = ttk.LabelFrame(parent, text="文本输入区域", padding="5")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # 文本输入框
        self.text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=15)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # 输入统计
        input_stats_frame = ttk.Frame(input_frame)
        input_stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.input_stats_label = ttk.Label(input_stats_frame, text="字符数: 0 | 行数: 0")
        self.input_stats_label.pack(side=tk.LEFT)

        # 快速操作按钮
        quick_frame = ttk.Frame(input_frame)
        quick_frame.pack(fill=tk.X)

        ttk.Button(quick_frame, text="示例数据", command=self.load_sample_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="清空", command=self.clear_input).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="粘贴", command=self.paste_from_clipboard).pack(side=tk.LEFT)

        # 历史记录面板
        history_frame = ttk.LabelFrame(parent, text="解析历史", padding="5")
        history_frame.pack(fill=tk.BOTH, expand=True)

        # 历史记录列表
        self.history_tree = ttk.Treeview(history_frame, columns=('时间', '数量'), show='headings', height=8)
        self.history_tree.heading('时间', text='解析时间')
        self.history_tree.heading('数量', text='记录数')
        self.history_tree.column('时间', width=120)
        self.history_tree.column('数量', width=60)

        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)

        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def create_preview_panel(self, parent):
        """创建预览面板"""
        # 结果预览区域
        preview_frame = ttk.LabelFrame(parent, text="解析结果预览", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        # 表格
        columns = ('序号', '收件人', '电话', '地址', '标签', '备注', '置信度', '创建时间')
        self.tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=20)

        # 设置列标题
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by_column(c))

        # 设置列宽
        self.tree.column('序号', width=50)
        self.tree.column('收件人', width=80)
        self.tree.column('电话', width=120)
        self.tree.column('地址', width=250)
        self.tree.column('标签', width=80)
        self.tree.column('备注', width=100)
        self.tree.column('置信度', width=60)
        self.tree.column('创建时间', width=120)

        # 滚动条
        tree_scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        # 放置表格和滚动条
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        # 表格操作按钮框架
        table_controls = ttk.Frame(preview_frame)
        table_controls.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Button(table_controls, text="全选", command=self.select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(table_controls, text="反选", command=self.invert_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(table_controls, text="编辑选中", command=self.edit_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(table_controls, text="删除选中", command=self.delete_selected).pack(side=tk.LEFT, padx=(0, 5))

        # 统计信息
        self.stats_label = ttk.Label(table_controls, text="总计: 0 条记录")
        self.stats_label.pack(side=tk.RIGHT)

    def create_status_bar(self):
        """创建状态栏"""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))

    def bind_events(self):
        """绑定事件"""
        # 窗口事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind('<Control-o>', lambda e: self.import_text())
        self.root.bind('<Control-e>', lambda e: self.export_excel())
        self.root.bind('<Control-l>', lambda e: self.clear_input())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<F5>', lambda e: self.parse_text())

        # 文本输入事件
        self.text_input.bind('<KeyRelease>', self.on_text_change)
        self.text_input.bind('<Control-a>', self.select_all_text)

        # 表格事件
        self.tree.bind('<Double-1>', self.on_item_double_click)
        self.tree.bind('<Button-3>', self.show_context_menu)
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_selection)

        # 历史记录事件
        self.history_tree.bind('<Double-1>', self.load_from_history)

    def load_window_state(self):
        """加载窗口状态"""
        try:
            geometry = self.config.get("window_geometry", "1400x900")
            self.root.geometry(geometry)

            # 加载历史记录
            self.load_parsing_history()

            # 加载最近的记录
            self.load_recent_records()

        except Exception as e:
            logging.error(f"Load window state failed: {e}")

    def start_auto_save(self):
        """启动自动保存"""
        if self.auto_save_var.get():
            self.root.after(300000, self.auto_save_data)  # 5分钟自动保存

    def auto_save_data(self):
        """自动保存数据"""
        try:
            if self.current_records:
                self.db.save_records_batch(self.current_records)
                logging.info("Auto save completed")
        except Exception as e:
            logging.error(f"Auto save failed: {e}")
        finally:
            if self.auto_save_var.get():
                self.root.after(300000, self.auto_save_data)

    # 文本处理方法
    def parse_text(self):
        """解析文本"""
        try:
            text = self.text_input.get(1.0, tk.END).strip()
            if not text:
                messagebox.showwarning("警告", "请输入要解析的文本")
                return

            self.status_var.set("正在解析...")
            self.progress.start()

            # 在新线程中执行解析
            threading.Thread(target=self._parse_text_thread, args=(text,), daemon=True).start()

        except Exception as e:
            self.progress.stop()
            self.status_var.set("解析失败")
            messagebox.showerror("错误", f"解析失败: {e}")
            logging.error(f"Parse text failed: {e}")

    def _parse_text_thread(self, text):
        """解析文本线程"""
        try:
            # 按行分割文本
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            records = []
            failed_lines = []

            for i, line in enumerate(lines):
                try:
                    record, metadata = AdvancedAddressExtractor.extract_info(line)
                    if record.name or record.phone or record.address:
                        records.append(record)
                    else:
                        failed_lines.append((i + 1, line))
                except Exception as e:
                    failed_lines.append((i + 1, line))
                    logging.warning(f"Failed to parse line {i + 1}: {e}")

            # 更新UI
            self.root.after(0, self._update_parse_results, records, failed_lines, text)

        except Exception as e:
            self.root.after(0, self._handle_parse_error, e)

    def _update_parse_results(self, records, failed_lines, original_text):
        """更新解析结果"""
        try:
            self.progress.stop()
            self.current_records = records
            self.filtered_records = records.copy()

            # 更新表格显示
            self.update_tree_display()

            # 保存到数据库
            if records:
                self.db.save_records_batch(records)
                self.db.save_parsing_history(original_text, len(records))

            # 更新历史记录
            self.load_parsing_history()

            # 更新状态
            success_count = len(records)
            total_lines = len([line for line in original_text.split('\n') if line.strip()])
            success_rate = (success_count / total_lines * 100) if total_lines > 0 else 0

            self.status_var.set(f"解析完成: 成功 {success_count} 条，成功率 {success_rate:.1f}%")

            # 显示失败的行
            if failed_lines:
                self.show_failed_lines(failed_lines)

            logging.info(f"Parse completed: {success_count} records extracted")

        except Exception as e:
            self.progress.stop()
            self.status_var.set("更新结果失败")
            logging.error(f"Update parse results failed: {e}")

    def _handle_parse_error(self, error):
        """处理解析错误"""
        self.progress.stop()
        self.status_var.set("解析失败")
        messagebox.showerror("错误", f"解析失败: {error}")

    def show_failed_lines(self, failed_lines):
        """显示解析失败的行"""
        if not failed_lines:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("解析失败的行")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=f"以下 {len(failed_lines)} 行解析失败:").pack(pady=10)

        # 失败行列表
        listbox = tk.Listbox(dialog, height=15)
        scrollbar = ttk.Scrollbar(dialog, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        for line_num, line_text in failed_lines:
            listbox.insert(tk.END, f"第{line_num}行: {line_text}")

        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0, 10))

        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=10)

    def update_tree_display(self):
        """更新表格显示"""
        # 清空现有项目
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 添加记录
        for i, record in enumerate(self.filtered_records, 1):
            confidence_str = f"{record.confidence:.2f}" if record.confidence else "0.00"
            create_time = record.created_time.split('T')[0] if record.created_time else ""

            self.tree.insert('', 'end', values=(
                i,
                record.name,
                record.phone,
                record.address,
                record.tags,
                record.notes,
                confidence_str,
                create_time
            ))

        # 更新统计信息
        self.stats_label.config(text=f"总计: {len(self.filtered_records)} 条记录")

    def clear_input(self):
        """清空输入"""
        self.text_input.delete(1.0, tk.END)
        self.update_input_stats()

    def clear_results(self):
        """清空结果"""
        self.current_records.clear()
        self.filtered_records.clear()
        self.update_tree_display()
        self.status_var.set("已清空结果")

    def clear_all(self):
        """清空所有"""
        if messagebox.askyesno("确认", "确定要清空所有数据吗？"):
            self.clear_input()
            self.clear_results()

    # 文本输入相关方法
    def on_text_change(self, event=None):
        """文本改变事件"""
        self.update_input_stats()

        # 自动解析
        if self.auto_parse_var.get():
            # 延迟解析，避免频繁触发
            if hasattr(self, '_parse_timer'):
                self.root.after_cancel(self._parse_timer)
            self._parse_timer = self.root.after(1000, self.parse_text)

    def update_input_stats(self):
        """更新输入统计"""
        text = self.text_input.get(1.0, tk.END)
        char_count = len(text.strip())
        line_count = len([line for line in text.split('\n') if line.strip()])
        self.input_stats_label.config(text=f"字符数: {char_count} | 行数: {line_count}")

    def load_sample_data(self):
        """加载示例数据"""
        sample_data = """张三 13812345678 北京市朝阳区建国路88号
李四,15987654321,上海市浦东新区陆家嘴金融贸易区
王五 18611112222 广州市天河区体育西路123号天河城
赵六，13688889999，深圳市南山区科技园南区
钱七 17755554444 杭州市西湖区文二路391号西湖国际科技大厦
孙八,13566667777,成都市锦江区红星路二段78号
周九 15433332222 武汉市武昌区中南路99号
吴十，18822221111，南京市鼓楼区中山路321号"""

        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(1.0, sample_data)
        self.update_input_stats()

    def paste_from_clipboard(self):
        """从剪贴板粘贴"""
        try:
            clipboard_text = self.root.clipboard_get()
            self.text_input.insert(tk.INSERT, clipboard_text)
            self.update_input_stats()
        except tk.TclError:
            messagebox.showwarning("警告", "剪贴板为空或无法访问")

    def select_all_text(self, event=None):
        """全选文本"""
        self.text_input.tag_add(tk.SEL, "1.0", tk.END)
        return 'break'

    # 表格操作方法
    def select_all(self):
        """全选表格项目"""
        for child in self.tree.get_children():
            self.tree.selection_add(child)

    def invert_selection(self):
        """反选"""
        selected = set(self.tree.selection())
        all_items = set(self.tree.get_children())
        new_selection = all_items - selected

        self.tree.selection_remove(*selected)
        self.tree.selection_add(*new_selection)

    def delete_selected(self):
        """删除选中项目"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要删除的记录")
            return

        if messagebox.askyesno("确认", f"确定要删除选中的 {len(selected)} 条记录吗？"):
            # 获取要删除的记录索引
            indices_to_remove = []
            for item in selected:
                index = self.tree.index(item)
                indices_to_remove.append(index)

            # 从后往前删除，避免索引变化
            indices_to_remove.sort(reverse=True)
            for index in indices_to_remove:
                if index < len(self.filtered_records):
                    del self.filtered_records[index]
                if index < len(self.current_records):
                    del self.current_records[index]

            self.update_tree_display()
            self.status_var.set(f"已删除 {len(selected)} 条记录")

    def edit_selected(self):
        """编辑选中项目"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要编辑的记录")
            return

        if len(selected) > 1:
            messagebox.showwarning("警告", "一次只能编辑一条记录")
            return

        item = selected[0]
        index = self.tree.index(item)
        if index < len(self.filtered_records):
            self.show_edit_dialog(self.filtered_records[index], index)

    def show_edit_dialog(self, record, index):
        """显示编辑对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("编辑记录")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()

        # 创建输入框
        fields = [
            ("收件人", record.name),
            ("电话", record.phone),
            ("地址", record.address),
            ("标签", record.tags),
            ("备注", record.notes)
        ]

        entries = {}
        for i, (label, value) in enumerate(fields):
            ttk.Label(dialog, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
            if label == "地址" or label == "备注":
                entry = scrolledtext.ScrolledText(dialog, height=3, width=40)
                entry.insert(1.0, value or "")
            else:
                entry = ttk.Entry(dialog, width=40)
                entry.insert(0, value or "")
            entry.grid(row=i, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))
            entries[label] = entry

        dialog.grid_columnconfigure(1, weight=1)

        # 按钮
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=len(fields), column=0, columnspan=2, pady=20)

        def save_changes():
            try:
                # 更新记录
                record.name = entries["收件人"].get()
                record.phone = entries["电话"].get()
                record.address = entries["地址"].get(1.0, tk.END).strip() if hasattr(entries["地址"], 'get') else entries["地址"].get()
                record.tags = entries["标签"].get()
                record.notes = entries["备注"].get(1.0, tk.END).strip() if hasattr(entries["备注"], 'get') else entries["备注"].get()
                record.updated_time = datetime.now().isoformat()

                # 更新显示
                self.update_tree_display()
                dialog.destroy()
                self.status_var.set("记录已更新")

            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e}")

        ttk.Button(button_frame, text="保存", command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def on_item_double_click(self, event):
        """表格项目双击事件"""
        self.edit_selected()

    def show_context_menu(self, event):
        """显示右键菜单"""
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="编辑", command=self.edit_selected)
        context_menu.add_command(label="删除", command=self.delete_selected)
        context_menu.add_separator()
        context_menu.add_command(label="复制电话", command=self.copy_phone)
        context_menu.add_command(label="复制地址", command=self.copy_address)
        context_menu.add_separator()
        context_menu.add_command(label="全选", command=self.select_all)
        context_menu.add_command(label="反选", command=self.invert_selection)

        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def copy_phone(self):
        """复制电话号码"""
        selected = self.tree.selection()
        if selected:
            item = selected[0]
            values = self.tree.item(item, 'values')
            if len(values) > 2 and values[2]:
                self.root.clipboard_clear()
                self.root.clipboard_append(values[2])
                self.status_var.set("已复制电话号码")

    def copy_address(self):
        """复制地址"""
        selected = self.tree.selection()
        if selected:
            item = selected[0]
            values = self.tree.item(item, 'values')
            if len(values) > 3 and values[3]:
                self.root.clipboard_clear()
                self.root.clipboard_append(values[3])
                self.status_var.set("已复制地址")

    def on_tree_selection(self, event):
        """表格选择改变事件"""
        selected = self.tree.selection()
        self.selected_records = []

        for item in selected:
            index = self.tree.index(item)
            if index < len(self.filtered_records):
                self.selected_records.append(self.filtered_records[index])

    def sort_by_column(self, column):
        """按列排序"""
        try:
            # 获取列数据
            data = []
            for child in self.tree.get_children():
                values = self.tree.item(child, 'values')
                data.append((values, child))

            # 确定排序列的索引
            columns = ('序号', '收件人', '电话', '地址', '标签', '备注', '置信度', '创建时间')
            col_index = columns.index(column)

            # 排序
            reverse = getattr(self, f'_{column}_sort_reverse', False)

            if column == '置信度':
                data.sort(key=lambda x: float(x[0][col_index]) if x[0][col_index] else 0, reverse=reverse)
            elif column == '序号':
                data.sort(key=lambda x: int(x[0][col_index]) if x[0][col_index].isdigit() else 0, reverse=reverse)
            else:
                data.sort(key=lambda x: x[0][col_index], reverse=reverse)

            # 重新排列树形控件中的项目
            for index, (values, child) in enumerate(data):
                self.tree.move(child, '', index)

            # 切换排序方向
            setattr(self, f'_{column}_sort_reverse', not reverse)

        except Exception as e:
            logging.error(f"Sort by column failed: {e}")

    # 搜索和筛选方法
    def on_search_change(self, event=None):
        """搜索改变事件"""
        keyword = self.search_var.get().strip()
        if keyword:
            self.filtered_records = [
                record for record in self.current_records
                if keyword.lower() in (record.name or "").lower() or
                   keyword.lower() in (record.phone or "").lower() or
                   keyword.lower() in (record.address or "").lower() or
                   keyword.lower() in (record.tags or "").lower()
            ]
        else:
            self.filtered_records = self.current_records.copy()

        self.apply_filter()
        self.update_tree_display()

    def on_filter_change(self, event=None):
        """筛选改变事件"""
        self.apply_filter()
        self.update_tree_display()

    def apply_filter(self):
        """应用筛选条件"""
        filter_type = self.filter_var.get()

        if filter_type == "全部":
            pass  # 不过滤
        elif filter_type == "有姓名":
            self.filtered_records = [r for r in self.filtered_records if r.name]
        elif filter_type == "有电话":
            self.filtered_records = [r for r in self.filtered_records if r.phone]
        elif filter_type == "有地址":
            self.filtered_records = [r for r in self.filtered_records if r.address]
        elif filter_type == "信息完整":
            self.filtered_records = [r for r in self.filtered_records if r.name and r.phone and r.address]
        elif filter_type == "信息不全":
            self.filtered_records = [r for r in self.filtered_records if not (r.name and r.phone and r.address)]

    # 文件操作方法
    def import_text(self):
        """导入文本文件"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择文本文件",
                filetypes=[
                    ("文本文件", "*.txt"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.text_input.delete(1.0, tk.END)
                self.text_input.insert(1.0, content)
                self.update_input_stats()
                self.status_var.set(f"已导入文件: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"导入文件失败: {e}")

    def import_excel(self):
        """导入Excel文件"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择Excel文件",
                filetypes=[
                    ("Excel文件", "*.xlsx *.xls"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                df = pd.read_excel(file_path)

                # 尝试识别列
                text_lines = []
                for _, row in df.iterrows():
                    line_parts = []
                    for value in row.values:
                        if pd.notna(value):
                            line_parts.append(str(value))
                    if line_parts:
                        text_lines.append(" ".join(line_parts))

                content = "\n".join(text_lines)
                self.text_input.delete(1.0, tk.END)
                self.text_input.insert(1.0, content)
                self.update_input_stats()
                self.status_var.set(f"已导入Excel: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"导入Excel失败: {e}")

    def export_excel(self):
        """导出Excel文件"""
        if not self.filtered_records:
            messagebox.showwarning("警告", "没有数据可以导出")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="保存Excel文件",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel文件", "*.xlsx"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                # 准备数据
                data = []
                for i, record in enumerate(self.filtered_records, 1):
                    data.append({
                        '序号': i,
                        '收件人': record.name,
                        '电话': record.phone,
                        '地址': record.address,
                        '标签': record.tags,
                        '备注': record.notes,
                        '置信度': record.confidence,
                        '创建时间': record.created_time,
                        '更新时间': record.updated_time
                    })

                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False, engine='openpyxl')

                self.status_var.set(f"已导出 {len(data)} 条记录到: {os.path.basename(file_path)}")
                messagebox.showinfo("成功", f"成功导出 {len(data)} 条记录")

        except Exception as e:
            messagebox.showerror("错误", f"导出Excel失败: {e}")

    def export_csv(self):
        """导出CSV文件"""
        if not self.filtered_records:
            messagebox.showwarning("警告", "没有数据可以导出")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="保存CSV文件",
                defaultextension=".csv",
                filetypes=[
                    ("CSV文件", "*.csv"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                data = []
                for i, record in enumerate(self.filtered_records, 1):
                    data.append({
                        '序号': i,
                        '收件人': record.name,
                        '电话': record.phone,
                        '地址': record.address,
                        '标签': record.tags,
                        '备注': record.notes,
                        '置信度': record.confidence,
                        '创建时间': record.created_time,
                        '更新时间': record.updated_time
                    })

                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

                self.status_var.set(f"已导出 {len(data)} 条记录到CSV文件")
                messagebox.showinfo("成功", f"成功导出 {len(data)} 条记录")

        except Exception as e:
            messagebox.showerror("错误", f"导出CSV失败: {e}")

    def export_json(self):
        """导出JSON文件"""
        if not self.filtered_records:
            messagebox.showwarning("警告", "没有数据可以导出")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="保存JSON文件",
                defaultextension=".json",
                filetypes=[
                    ("JSON文件", "*.json"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                data = [record.to_dict() for record in self.filtered_records]

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                self.status_var.set(f"已导出 {len(data)} 条记录到JSON文件")
                messagebox.showinfo("成功", f"成功导出 {len(
