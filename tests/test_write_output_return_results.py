from pathlib import Path

import numpy as np

from sfincs_jax.io import write_sfincs_jax_output_h5


def test_write_output_return_results(tmp_path: Path) -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme4_1species_tiny.input.namelist"
    out_path = tmp_path / "sfincsOutput.h5"

    resolved, results = write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        return_results=True,
    )

    assert resolved == out_path.resolve()
    assert resolved.exists()
    assert isinstance(results, dict)
    assert "Ntheta" in results
    assert int(np.asarray(results["Ntheta"]).reshape(())) > 0
