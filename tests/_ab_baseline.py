"""The pre-#189 behaviour, spelled out: every knob the browser A/B batches
added, at the value it had before. The batch tests start from this so a
"knob off" test keeps pinning the old behaviour whatever the shipped default."""

BASELINE = dict(
    clearance_capture_bg=False, robots_persona_headers=False, sidecar_fast_boot=False,
    swarm_read_via_engine=False, parallel_tools=False, read_paging=False,
    pdf_fast_extract=False, engine_render_quiesce=0, engine_page_referer=False,
    engine_binary_body=False, engine_render_prelude=False, engine_script_pipeline=False,
    meta_refresh_strict=False,
    charset_detect=False,
    http_keepalive_s=5.0,
    engine_live_dom=False,
    engine_tls_impersonate=False,
    thin_heavy_rescue=False,
    engine_fetch_pool=False,
    engine_modules=False,
    engine_code_cache=False,
    engine_render_quiet_ms=0,
    engine_isolates=1,
    engine_render_rung=False,
    extract_youtube_lists=False, extract_x_posts=False, listing_packages=False,
    items_vehicle_listings=False,
)
