import os
import tempfile

def main() -> None:
    # Force a cache miss so the subprocess inspection path under test is used,
    # while ensuring the generated cache is removed afterward.
    with tempfile.TemporaryDirectory(prefix="vllm-model-inspection-") as cache:
        os.environ["VLLM_CACHE_ROOT"] = cache
        import vllm.model_executor.models.registry as registry

        architecture = "LlamaForCausalLM"
        model = registry.ModelRegistry.models[architecture]
        print("Model object:", type(model), model)
        print("About to call _try_inspect_model_cls…")
        result = registry._try_inspect_model_cls(architecture, model)
        if result is None:
            raise AssertionError("vLLM model inspection returned None")
        print("PASS returned:", result)


if __name__ == "__main__":
    main()
