"""Sliders are discovered from a .frag's own uniform declarations.

Annotation is opt-in: an unannotated uniform yields no ShaderParam, so a
shader written before this existed still compiles and still runs, just with
no UI. That property is what keeps march.frag working.
"""
from services.shader_params import parse_shader_params, ShaderParam


def names(src):
    return [p.name for p in parse_shader_params(src)]


def one(src) -> ShaderParam:
    params = parse_shader_params(src)
    assert len(params) == 1, params
    return params[0]


def test_an_unannotated_uniform_yields_nothing():
    assert names("uniform float speed;") == []


def test_a_shader_with_no_uniforms_yields_nothing():
    assert names("void main(){ fragColor = vec4(0.0); }") == []


def test_a_ranged_float():
    p = one('uniform float speed;   // 0..5 = 1.5   "Speed"')
    assert (p.name, p.kind, p.lo, p.hi) == ("speed", "float", 0.0, 5.0)
    assert p.default == (1.5,)
    assert p.label == "Speed"
    assert p.components == 1


def test_a_ranged_int():
    p = one('uniform int octaves; // 1..8 = 4 "Octaves"')
    assert (p.name, p.kind, p.lo, p.hi, p.default) == ("octaves", "int", 1.0, 8.0, (4.0,))


def test_a_bool_has_no_range():
    p = one('uniform bool invert; // = false "Invert"')
    assert (p.name, p.kind, p.default) == ("invert", "bool", (0.0,))


def test_a_bool_defaulting_true():
    assert one('uniform bool on; // = true "On"').default == (1.0,)


def test_a_colour_vec3():
    p = one('uniform vec3 tint; // color = 1,.5,0 "Tint"')
    assert (p.name, p.kind, p.components) == ("tint", "color", 3)
    assert p.default == (1.0, 0.5, 0.0)


def test_a_ranged_vec2_is_a_multi_slider():
    p = one('uniform vec2 centre; // 0..1 = .5,.5 "Centre"')
    assert (p.name, p.kind, p.components, p.default) == ("centre", "vec", 2, (0.5, 0.5))


def test_the_label_defaults_to_a_titled_name():
    assert one("uniform float draw_speed; // 0..1 = 0.5").label == "Draw Speed"


def test_declarations_are_found_among_real_shader_text():
    src = (
        "#version 430\n"
        "in vec2 texcoord;\n"
        "out vec4 fragColor;\n"
        "uniform vec2 canvas_resolution;\n"
        'uniform float speed;  // 0..5 = 1.0 "Speed"\n'
        'uniform int   steps;  // 1..64 = 8 "Steps"\n'
        "void main(){ fragColor = vec4(speed); }\n"
    )
    assert names(src) == ["speed", "steps"]


def test_a_commented_out_declaration_is_ignored():
    assert names('// uniform float speed; // 0..5 = 1.0 "Speed"') == []


def test_a_malformed_annotation_is_ignored_rather_than_raising():
    assert names('uniform float speed; // nonsense here') == []
