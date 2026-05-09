"""Sandbox abstraction — multi-provider execution isolation.

Providers:
  - local: execute in the host process/terminal (default)
  - docker: execute in an isolated Docker container
  - ssh: execute on a remote machine via SSH
  - cloud: execute in a cloud sandbox (Daytona, etc.)
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SandboxProvider(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"
    SSH = "ssh"
    CLOUD = "cloud"


@dataclass
class SandboxResult:
    """Result of a sandbox execution."""
    output: str
    exit_code: int
    duration_ms: float
    provider: str
    sandbox_id: str = ""


@dataclass
class SandboxConfig:
    """Configuration for a sandbox provider."""
    provider: SandboxProvider = SandboxProvider.LOCAL
    timeout: float = 300.0
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    # Docker-specific
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"

    # SSH-specific
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_key_path: str = ""

    # Cloud-specific
    cloud_api_key: str = ""
    cloud_endpoint: str = ""


class Sandbox(ABC):
    """Abstract sandbox interface."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._id = f"{config.provider.value}_{int(time.time() * 1000)}"

    @property
    def sandbox_id(self) -> str:
        return self._id

    @abstractmethod
    def execute(self, command: str, timeout: float | None = None) -> SandboxResult:
        """Execute a command in the sandbox and return the result."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the sandbox provider is available."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        ...


class LocalSandbox(Sandbox):
    """Execute commands directly on the host."""

    def __init__(self, config: SandboxConfig | None = None):
        super().__init__(config or SandboxConfig(provider=SandboxProvider.LOCAL))

    def execute(self, command: str, timeout: float | None = None) -> SandboxResult:
        start = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout,
                cwd=self.config.workdir,
                env={**__import__("os").environ, **self.config.env},
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            return SandboxResult(
                output=output.strip(),
                exit_code=result.returncode,
                duration_ms=(time.time() - start) * 1000,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                output="Command timed out.",
                exit_code=-1,
                duration_ms=(time.time() - start) * 1000,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )

    def is_available(self) -> bool:
        return True  # Always available

    def cleanup(self) -> None:
        pass  # No cleanup needed


class DockerSandbox(Sandbox):
    """Execute commands in an isolated Docker container."""

    def __init__(self, config: SandboxConfig | None = None):
        super().__init__(config or SandboxConfig(provider=SandboxProvider.DOCKER))
        self._container_name = f"kairos-sandbox-{self._id}"

    def execute(self, command: str, timeout: float | None = None) -> SandboxResult:
        if not self.is_available():
            return SandboxResult(
                output="Docker is not available.",
                exit_code=-1,
                duration_ms=0,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )

        docker_cmd = [
            "docker", "run", "--rm",
            "--name", self._container_name,
            "--network", self.config.docker_network,
            "--workdir", self.config.workdir or "/workspace",
        ]
        for k, v in self.config.env.items():
            docker_cmd.extend(["-e", f"{k}={v}"])
        docker_cmd.append(self.config.docker_image)
        docker_cmd.extend(["bash", "-c", command])

        start = time.time()
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            return SandboxResult(
                output=output.strip(),
                exit_code=result.returncode,
                duration_ms=(time.time() - start) * 1000,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )
        except subprocess.TimeoutExpired:
            # Clean up the container
            subprocess.run(["docker", "rm", "-f", self._container_name], capture_output=True)
            return SandboxResult(
                output="Command timed out.",
                exit_code=-1,
                duration_ms=(time.time() - start) * 1000,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def cleanup(self) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self._container_name],
            capture_output=True,
        )


class SSHSandbox(Sandbox):
    """Execute commands on a remote machine via SSH."""

    def __init__(self, config: SandboxConfig | None = None):
        super().__init__(config or SandboxConfig(provider=SandboxProvider.SSH))

    def execute(self, command: str, timeout: float | None = None) -> SandboxResult:
        if not self.config.ssh_host:
            return SandboxResult(
                output="SSH host not configured.",
                exit_code=-1,
                duration_ms=0,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(self.config.ssh_port),
        ]
        if self.config.ssh_key_path:
            ssh_cmd.extend(["-i", self.config.ssh_key_path])

        target = f"{self.config.ssh_user}@{self.config.ssh_host}" if self.config.ssh_user else self.config.ssh_host
        ssh_cmd.append(target)

        # Build the remote command with env and workdir
        remote_cmd = ""
        if self.config.workdir:
            remote_cmd += f"cd {self.config.workdir} && "
        for k, v in self.config.env.items():
            remote_cmd += f"{k}={v} "
        remote_cmd += command

        ssh_cmd.append(remote_cmd)

        start = time.time()
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            return SandboxResult(
                output=output.strip(),
                exit_code=result.returncode,
                duration_ms=(time.time() - start) * 1000,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                output="SSH command timed out.",
                exit_code=-1,
                duration_ms=(time.time() - start) * 1000,
                provider=self.config.provider.value,
                sandbox_id=self._id,
            )

    def is_available(self) -> bool:
        if not self.config.ssh_host:
            return False
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                 f"{self.config.ssh_user}@{self.config.ssh_host}" if self.config.ssh_user else self.config.ssh_host,
                 "echo ok"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def cleanup(self) -> None:
        pass  # SSH connections are ephemeral


def create_sandbox(config: SandboxConfig) -> Sandbox:
    """Factory function to create a sandbox from configuration."""
    providers = {
        SandboxProvider.LOCAL: LocalSandbox,
        SandboxProvider.DOCKER: DockerSandbox,
        SandboxProvider.SSH: SSHSandbox,
    }
    provider_cls = providers.get(config.provider)
    if not provider_cls:
        raise ValueError(f"Unknown sandbox provider: {config.provider}")
    return provider_cls(config)
