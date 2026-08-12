from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=300)
at.run()
print("no-upload run, exceptions:", at.exception)
assert not at.exception

# switch to folder mode and rerun through all tabs
at.sidebar.radio[0].set_value("Read from folder").run()
print("folder mode exceptions:", at.exception)
if at.exception:
    for e in at.exception:
        print(e.value)
        print(e.stack_trace)
assert not at.exception
print("markdown blocks:", len(at.markdown), "dataframes:", len(at.dataframe), "metrics:", len(at.metric))
for m in at.metric[:8]:
    print("  METRIC", m.label, "=", m.value)
for w in at.warning: print("  WARN:", w.value[:120])
for e in at.error: print("  ERROR:", e.value[:160])
print("OK")
