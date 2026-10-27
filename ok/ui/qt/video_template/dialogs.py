from PySide6.QtWidgets import QApplication, QPlainTextEdit
from qfluentwidgets import BodyLabel, FluentIcon, LineEdit, MessageBoxBase, PrimaryPushButton, SubtitleLabel

from ok.ui.qt.util.Alert import alert_info


class ExportTemplateConflictDialog(MessageBoxBase):
    def __init__(self, parent_page, conflicts, source_path, target_name):
        super().__init__(parent_page.window() if parent_page else None)
        self.apply_to_session = False
        self.titleLabel = SubtitleLabel(parent_page.tr("发现同名模板标注"), self)
        self.messageLabel = BodyLabel(
            parent_page.tr("已有模板标注与本次导出的标注同名。确认后会覆盖这些同名标注。"),
            self,
        )
        self.detailEdit = QPlainTextEdit(self)
        self.detailEdit.setReadOnly(True)
        self.detailEdit.setMinimumHeight(300)
        self.detailEdit.setPlainText(
            parent_page.export_conflict_detail_text(conflicts, source_path, target_name)
        )
        self.applyButton = PrimaryPushButton(
            FluentIcon.SAVE,
            parent_page.tr("确定且本次会话不再询问"),
            self,
        )

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.messageLabel)
        self.viewLayout.addWidget(self.detailEdit)
        self.viewLayout.addWidget(self.applyButton)
        self.yesButton.setText(parent_page.tr("确定覆盖"))
        self.cancelButton.setText(parent_page.tr("取消"))
        self.widget.setMinimumWidth(720)
        self.applyButton.clicked.connect(self.accept_for_session)

    def accept_for_session(self):
        self.apply_to_session = True
        self.accept()


class AgentAssistPromptDialog(MessageBoxBase):
    def __init__(self, parent_page, context):
        super().__init__(parent_page.window() if parent_page else None)
        self.parent_page = parent_page
        self.context = dict(context or {})
        self.titleLabel = SubtitleLabel(parent_page.tr("Agent 辅助加工标注"), self)
        self.targetLabel = BodyLabel(parent_page.tr("希望 Agent 识别什么画面"), self)
        self.targetEdit = LineEdit(self)
        self.targetEdit.setPlaceholderText(parent_page.tr("例如：所有空农田、可点击的任务图标、成熟小麦格子"))
        self.promptEdit = QPlainTextEdit(self)
        self.promptEdit.setReadOnly(True)
        self.promptEdit.setMinimumHeight(320)
        self.copyButton = PrimaryPushButton(FluentIcon.COPY, parent_page.tr("复制提示词"), self)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.targetLabel)
        self.viewLayout.addWidget(self.targetEdit)
        self.viewLayout.addWidget(self.promptEdit)
        self.viewLayout.addWidget(self.copyButton)
        self.yesButton.setText(parent_page.tr("关闭"))
        self.cancelButton.setVisible(False)
        self.widget.setMinimumWidth(680)

        self.targetEdit.textChanged.connect(self.refresh_prompt)
        self.copyButton.clicked.connect(self.copy_prompt)
        self.refresh_prompt()

    def refresh_prompt(self, *_args):
        self.promptEdit.setPlainText(
            self.parent_page.build_agent_assist_prompt(self.context, self.targetEdit.text())
        )

    def copy_prompt(self):
        QApplication.clipboard().setText(self.promptEdit.toPlainText())
        alert_info(self.parent_page.tr("已复制 Agent 辅助提示词"))
