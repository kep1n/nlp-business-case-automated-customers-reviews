import logging
import logging.handlers
import socket
import json
from logging.handlers import RotatingFileHandler


class HostnameFilter(logging.Filter):
    def __init__(self):
        """
        Initializes the HostnameFilter instance.
        Sets the hostname and environment attributes based on the machine's hostname.
        """
        super().__init__()
        self.hostname = socket.gethostname()
        # self.environment = self.get_environment()

    def filter(self, record):
        """
        Adds the hostname and environment to the log record.

        Parameters
        ----------
            record (logging.LogRecord): The log record to modify.

        Returns
        -------
            bool: True to allow the record to be processed, False to discard it.
        """
        record.hostname = self.hostname
        # record.environment = self.environment
        return True

    def get_environment(self) -> str:
        """
        Returns the environment of the machine running the script based on its hostname.

        The environment is determined by the prefix of the hostname:
        - 'p-' for production (PRD)
        - 's-' for staging (STG)
        - 'd-' for development (DEV)
        - 'Unknown' if the hostname does not match any of the above patterns

        Returns:
            str: The environment of the machine running the script.
        """
        if 'Kep' in self.hostname.upper():
            return 'TEST'
        else:
            return 'PRD'


class JsonFormatter(logging.Formatter):
    """
    A custom logging formatter that outputs log records in JSON format.

    This formatter includes the timestamp, hostname, environment, app name, log level, and message in the JSON output.
    """
    def format(self, record):
        """
        Format a log record as a JSON string.

        Parameters
        ----------
            record: The log record to format
        Returns
        -------
            str: A JSON string representing the log record
        """
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "hostname": getattr(record, "hostname", "unknown"),
            # "environment": getattr(record, "environment", "unknown"),
            "app_name": getattr(record, "app_name", "unknown"),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record, ensure_ascii=False)


class AppFilter(logging.Filter):

    def __init__(self, app_name):
        super().__init__()
        self.app_name = app_name

    def filter(self, record):
        record.app_name = self.app_name  # Agrega la variable app_name al registro de log
        return True  # Devuelve True para permitir que el log pase


class ProjectLogger:
    def __init__(self, log_path, app_name='Project Lab 3'):
        """
        Inicializa el logger para enviar mensajes a un servidor Syslog.
        :param log_path: Ruta del archivo de log.
        :param app_name: Nombre de la aplicación.
        """
        self.logger = logging.getLogger('ProjectLogger')
        self.logger.setLevel(logging.INFO)  # Nivel de log

        formatter = JsonFormatter()

        # Agregar un StreamHandler para ver los mensajes en la consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(HostnameFilter())
        console_handler.addFilter(AppFilter(app_name))
        self.logger.addHandler(console_handler)

        rotating_handler = RotatingFileHandler(log_path, maxBytes=10485760, backupCount=1)
        rotating_handler.setFormatter(formatter)
        rotating_handler.addFilter(HostnameFilter())
        rotating_handler.addFilter(AppFilter(app_name))
        self.logger.addHandler(rotating_handler)

    def get_logger(self):
        return self.logger
