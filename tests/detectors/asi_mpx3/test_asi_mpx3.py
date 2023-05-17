from libertem_live.api import LiveContext
from libertem.udf.sum import SumUDF


def test_smoke(ctx_pipelined: LiveContext):
    import requests
    # configure manually for now (this should be done in the active mode later):
    requests.put("http://localhost:8080/server/destination/", json={
        "Image": [{
            "Base": "tcp://listen@localhost:8284",
            "FilePattern": "f%Hms_",
            "Format": "pgm",
            "Mode": "count",
        }]
    })
    config = requests.get("http://localhost:8080/detector/config").json()
    config['nTriggers'] = 128 * 128
    config['TriggerPeriod'] = config['ExposureTime'] = 0.0005
    requests.put("http://localhost:8080/detector/config/", json=config)
    requests.get("http://localhost:8080/measurement/start/")

    with ctx_pipelined.make_connection('asi_mpx3').open(
        data_port=8284,
    ) as conn:
        pending_aq = conn.wait_for_acquisition(3.0)
        assert pending_aq is not None
        aq = ctx_pipelined.make_acquisition(
            conn=conn,
            pending_aq=pending_aq,
        )
        ctx_pipelined.run_udf(dataset=aq, udf=SumUDF())
