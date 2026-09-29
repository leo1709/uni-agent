from uni_agent.tasks.base import Task, TaskResult
from uni_agent.tasks.registry import register_task


@register_task("rollout_smoke")
class RolloutSmokeTask(Task):
    async def run(self) -> TaskResult:
        agent = self.build_agent()

        async with self.build_sandbox() as sandbox:
            result = await agent.run(
                sandbox=sandbox,
                messages=self.config.prompt,
                workdir="/tmp",
            )

        marker_observed = False
        for message in result.transcript:
            if message.get("role") == "tool":
                content = str(message.get("content", ""))

                marker = "UNI_AGENT_VERL_SMOKE_OK"
                if marker in content:
                    marker_observed = True
                    break

        passed = (
            result.finished
            and marker_observed
            and result.info.get("errors") == 0
            and result.info.get("timeouts") == 0
        )

        return TaskResult(
            reward=float(passed),
            accuracy=float(passed),
            finished=result.finished,
            extra_info={
                "passed": passed,
                "marker_observed": marker_observed,
                "agent_info": result.info,
                "transcript": result.transcript,
            },
        )