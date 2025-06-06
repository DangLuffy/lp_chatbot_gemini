# /app/solver/geometric_solver.py

import numpy as np
import itertools
import logging
from typing import Dict, List, Any, Tuple, Optional

# Thư viện để vẽ đồ thị và xử lý ảnh
import matplotlib
matplotlib.use('Agg') # Chuyển sang backend không hiển thị UI, cần thiết cho server
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

# Hàm _normalize_problem_data không còn cần thiết vì solver sẽ nhận trực tiếp "Định dạng A"
# Chúng ta sẽ xử lý việc đọc "Định dạng A" trong hàm solve_with_geometric_method

def _plot_feasible_region(
    problem_title:str,
    constraints_to_draw: List[Dict[str, Any]], # Mong đợi 'coefficients', 'type', 'rhs', 'name'
    feasible_vertices: List[np.ndarray],
    optimal_vertex: Optional[np.ndarray] = None,
    is_unbounded: bool = False,
    variable_names: List[str] = ["x1", "x2"] # Thêm tên biến để hiển thị trục
) -> str:
    fig, ax = plt.subplots(figsize=(9, 9))

    plot_main_title = problem_title
    if is_unbounded:
        plot_subtitle = "Miền Khả Thi (Có Thể Không Bị Chặn)"
    elif not feasible_vertices and constraints_to_draw:
        plot_subtitle = "Không có Đỉnh Khả Thi (Miền rỗng, nửa mặt phẳng, hoặc không bị chặn)"
    elif not constraints_to_draw:
        plot_subtitle = "Không có ràng buộc nào được cung cấp"
    else:
        plot_subtitle = "Miền Khả Thi và Điểm Tối Ư"

    fig.suptitle(plot_main_title, fontsize=16, fontweight='bold')
    ax.set_title(plot_subtitle, fontsize=12)

    if not is_unbounded and feasible_vertices and len(feasible_vertices) >= 3:
        center = np.mean(feasible_vertices, axis=0)
        # Sắp xếp các đỉnh để vẽ đa giác lồi
        sorted_vertices = sorted(feasible_vertices, key=lambda v: np.arctan2(v[1] - center[1], v[0] - center[0]))
        polygon = plt.Polygon(sorted_vertices, color='lightcyan', alpha=0.7, ec='teal', linewidth=1.5)
        ax.add_patch(polygon)
    elif not is_unbounded and feasible_vertices and len(feasible_vertices) == 2: # Đoạn thẳng
        ax.plot([v[0] for v in feasible_vertices], [v[1] for v in feasible_vertices], color='teal', linewidth=3, alpha=0.7)
    elif not is_unbounded and feasible_vertices and len(feasible_vertices) == 1: # Một điểm
         ax.plot(feasible_vertices[0][0], feasible_vertices[0][1], 'o', color='teal', markersize=8, alpha=0.7)

    # Tính toán giới hạn cho trục vẽ
    padding_factor = 0.3
    x_coords_for_lim = [0.0] # Luôn bao gồm gốc tọa độ
    y_coords_for_lim = [0.0]

    if feasible_vertices:
        x_coords_for_lim.extend([v[0] for v in feasible_vertices])
        y_coords_for_lim.extend([v[1] for v in feasible_vertices])
    if optimal_vertex is not None and not is_unbounded:
        x_coords_for_lim.append(optimal_vertex[0])
        y_coords_for_lim.append(optimal_vertex[1])

    # Thêm các điểm cắt trục của ràng buộc vào tính toán giới hạn
    for constr in constraints_to_draw:
        c = constr.get("coefficients", [0,0]) # Đã là 'coefficients' từ solve_with_geometric_method
        r = constr.get("rhs", 0)
        if len(c) == 2:
            if abs(c[1]) > 1e-6 and abs(c[0]) > 1e-6 : # Giao với trục x (y=0)
                x_coords_for_lim.append(r/c[0])
            if abs(c[0]) > 1e-6 and abs(c[1]) > 1e-6: # Giao với trục y (x=0)
                y_coords_for_lim.append(r/c[1])
            elif abs(c[1]) < 1e-6 and abs(c[0]) > 1e-6: # Đường thẳng đứng x = r/c[0]
                x_coords_for_lim.append(r/c[0])
            elif abs(c[0]) < 1e-6 and abs(c[1]) > 1e-6: # Đường nằm ngang y = r/c[1]
                y_coords_for_lim.append(r/c[1])


    if not x_coords_for_lim or len(x_coords_for_lim) <= 1: x_coords_for_lim.extend([-5.0, 5.0]) # Mặc định nếu không có điểm nào
    if not y_coords_for_lim or len(y_coords_for_lim) <= 1: y_coords_for_lim.extend([-5.0, 5.0])

    x_min_lim, x_max_lim = min(x_coords_for_lim), max(x_coords_for_lim)
    y_min_lim, y_max_lim = min(y_coords_for_lim), max(y_coords_for_lim)

    x_current_range = (x_max_lim - x_min_lim) if (x_max_lim - x_min_lim) > 1e-1 else 10.0 # Mở rộng nếu range quá nhỏ
    y_current_range = (y_max_lim - y_min_lim) if (y_max_lim - y_min_lim) > 1e-1 else 10.0

    ax.set_xlim(x_min_lim - x_current_range * padding_factor - 1, x_max_lim + x_current_range * padding_factor + 1)
    ax.set_ylim(y_min_lim - y_current_range * padding_factor - 1, y_max_lim + y_current_range * padding_factor + 1)

    line_x_coords_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)

    cmap = plt.cm.get_cmap('viridis') # Sử dụng colormap khác
    num_total_colors = cmap.N if hasattr(cmap, 'N') else 10

    for i, constr in enumerate(constraints_to_draw):
        c = constr["coefficients"] # Đã là 'coefficients'
        r = constr["rhs"]
        op = constr["type"]       # Đã là 'type'
        constr_name = constr.get("name", f"({i+1})")
        current_line_color = cmap(i / max(1, len(constraints_to_draw) -1 ) if len(constraints_to_draw)>1 else 0.5)


        line_drawn = False
        mid_point_for_arrow = None

        if abs(c[1]) > 1e-6: # Đường không thẳng đứng
            y_vals_line = (r - c[0] * line_x_coords_plot) / c[1]
            line, = ax.plot(line_x_coords_plot, y_vals_line, label=constr_name, alpha=0.9, linestyle='-', color=current_line_color, linewidth=1.5)
            line_drawn = True
            visible_indices = np.where((y_vals_line >= ax.get_ylim()[0]) & (y_vals_line <= ax.get_ylim()[1]))[0]
            if len(visible_indices) > 0:
                mid_idx = visible_indices[len(visible_indices) // 2]
                mid_point_for_arrow = (line_x_coords_plot[mid_idx], y_vals_line[mid_idx])
        elif abs(c[0]) > 1e-6: # Đường thẳng đứng
            x_val_line = r / c[0]
            y_coords_for_vertical_line = np.array([ax.get_ylim()[0], ax.get_ylim()[1]])
            line, = ax.plot([x_val_line, x_val_line], y_coords_for_vertical_line, label=constr_name, alpha=0.9, linestyle='-', color=current_line_color, linewidth=1.5)
            line_drawn = True
            mid_point_for_arrow = (x_val_line, np.mean(y_coords_for_vertical_line))

        if line_drawn and mid_point_for_arrow:
            arrow_color_to_use = line.get_color()
            normal_vector = np.array([c[0], c[1]])
            norm = np.linalg.norm(normal_vector)
            if norm < 1e-6: continue
            normal_vector_unit = normal_vector / norm

            plot_diag_len = np.sqrt((ax.get_xlim()[1]-ax.get_xlim()[0])**2 + (ax.get_ylim()[1]-ax.get_ylim()[0])**2)
            actual_arrow_length = plot_diag_len / 20.0 # Điều chỉnh độ dài mũi tên

            direction_sign = 0
            if op == "<=" or op == "≤": direction_sign = -1
            elif op == ">=" or op == "≥": direction_sign = 1

            if direction_sign != 0:
                mid_x, mid_y = mid_point_for_arrow
                arrow_dir_x_comp = direction_sign * normal_vector_unit[0]
                arrow_dir_y_comp = direction_sign * normal_vector_unit[1]

                base_offset_len = actual_arrow_length * 0.1
                arrow_base_x = mid_x + arrow_dir_x_comp * base_offset_len
                arrow_base_y = mid_y + arrow_dir_y_comp * base_offset_len
                arrow_tip_x = arrow_base_x + arrow_dir_x_comp * actual_arrow_length
                arrow_tip_y = arrow_base_y + arrow_dir_y_comp * actual_arrow_length

                ax.annotate("", xy=(arrow_tip_x, arrow_tip_y),
                            xytext=(arrow_base_x, arrow_base_y),
                            arrowprops=dict(arrowstyle="-|>", color=arrow_color_to_use, lw=1.2, mutation_scale=15))

                text_offset_from_arrow_body = actual_arrow_length * 0.4
                text_x = arrow_tip_x + arrow_dir_x_comp * text_offset_from_arrow_body
                text_y = arrow_tip_y + arrow_dir_y_comp * text_offset_from_arrow_body

                ax.text(text_x, text_y, constr_name, fontsize=7, color=arrow_color_to_use,
                        ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.15', fc='white', alpha=0.7, ec='none'))

    current_x_range_plot = ax.get_xlim()[1] - ax.get_xlim()[0]
    current_y_range_plot = ax.get_ylim()[1] - ax.get_ylim()[0]

    if feasible_vertices:
        for i, v_point in enumerate(feasible_vertices):
            ax.plot(v_point[0], v_point[1], 'o', color='crimson', markersize=7, label='Đỉnh Khả Thi' if i==0 and not is_unbounded else None, zorder=5)
            ax.text(v_point[0] + 0.015 * current_x_range_plot, v_point[1] + 0.015 * current_y_range_plot, f'{chr(65+i)} ({v_point[0]:.2f}, {v_point[1]:.2f})', fontsize=8, zorder=6)

    if optimal_vertex is not None and not is_unbounded:
        ax.plot(optimal_vertex[0], optimal_vertex[1], 'o', color='gold', markersize=12, mec='black', label='Điểm Tối Ư', zorder=7)
        ax.text(optimal_vertex[0] + 0.015 * current_x_range_plot, optimal_vertex[1] - 0.04 * current_y_range_plot, f'Tối ưu\n({optimal_vertex[0]:.2f}, {optimal_vertex[1]:.2f})', fontsize=9, color='darkgreen', fontweight='bold', zorder=8)

    ax.axhline(0, color='black', linewidth=0.7, zorder=0)
    ax.axvline(0, color='black', linewidth=0.7, zorder=0)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray', zorder=-1)

    # Sử dụng tên biến được truyền vào
    ax.set_xlabel(f"${variable_names[0]}$", fontsize=14)
    ax.set_ylabel(f"${variable_names[1]}$", fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique_labels_dict = {}
        for handle, label_text in zip(handles, labels):
            if label_text not in unique_labels_dict:
                unique_labels_dict[label_text] = handle
        ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(), fontsize=9, loc='upper right', framealpha=0.7)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


def solve_with_geometric_method(problem_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Giải bài toán LP 2 biến bằng phương pháp hình học.
    Hàm này giờ đây mong đợi problem_data ở "Định dạng A".
    """
    logs = ["--- Starting Geometric Solver with Format A input ---"]

    # 1. Đọc và kiểm tra dữ liệu đầu vào từ "Định dạng A"
    objective_type = problem_data.get("objective", "maximize").lower()
    objective_coeffs_list = problem_data.get("coeffs", [])
    variable_names = problem_data.get("variables_names_for_title_only", [])
    constraints_input_format_A = problem_data.get("constraints", [])

    if not all([objective_type, isinstance(objective_coeffs_list, list), isinstance(variable_names, list), isinstance(constraints_input_format_A, list)]):
        logs.append("ERROR: Invalid or missing basic fields in problem_data (objective, coeffs, variables_names_for_title_only, constraints).")
        logger.error("Invalid or missing basic fields for Format A.")
        # Tạo plot lỗi cơ bản
        plot_base64_error_input = _plot_feasible_region("Lỗi: Dữ liệu đầu vào không hợp lệ", [], [], None, is_unbounded=True)
        return {"status": "Error", "message": "Invalid input data structure.", "plot_image_base64": plot_base64_error_input}, logs


    if len(variable_names) != 2:
        error_msg = "Geometric method only supports problems with exactly 2 variables."
        logs.append(f"ERROR: {error_msg} Received {len(variable_names)} variables: {variable_names}")
        logger.error(error_msg)
        # Tạo plot lỗi với tên biến nếu có
        title_for_error_plot = f"Lỗi: Chỉ hỗ trợ 2 biến (Nhận được: {', '.join(variable_names[:2])}{'...' if len(variable_names)>2 else ''})"
        plot_base64_error_vars = _plot_feasible_region(title_for_error_plot, [], [], None, is_unbounded=True, variable_names=variable_names[:2] if variable_names else ["x1","x2"])
        return {"status": "Error", "message": error_msg, "plot_image_base64": plot_base64_error_vars}, logs

    if len(objective_coeffs_list) != 2:
        error_msg = f"Geometric method expects 2 objective coefficients for 2 variables, got {len(objective_coeffs_list)}."
        logs.append(f"ERROR: {error_msg}")
        logger.error(error_msg)
        plot_base64_error_coeffs = _plot_feasible_region(f"Lỗi: Số hệ số mục tiêu không khớp ({len(objective_coeffs_list)} thay vì 2)", [], [], None, is_unbounded=True, variable_names=variable_names)
        return {"status": "Error", "message": error_msg, "plot_image_base64": plot_base64_error_coeffs}, logs

    # Tạo tiêu đề cho đồ thị
    obj_expr_terms = []
    for i, coeff_val in enumerate(objective_coeffs_list):
        var_name_display = variable_names[i]
        term_str = ""
        if abs(coeff_val) == 1:
             term_str = f"{'-' if coeff_val < 0 else ''}{var_name_display}"
        elif coeff_val != 0:
             term_str = f"{coeff_val:g}{var_name_display}" # :g để bỏ số 0 thừa

        if term_str:
            if obj_expr_terms and coeff_val > 0 and not obj_expr_terms[-1].endswith("- "): # Thêm dấu + nếu cần
                obj_expr_terms.append(f"+ {term_str.lstrip('+')}")
            elif obj_expr_terms and coeff_val < 0 :
                 obj_expr_terms.append(f" {term_str}")
            else: # Số hạng đầu tiên
                obj_expr_terms.append(term_str)

    problem_title_obj_part = "".join(obj_expr_terms).strip()
    if not problem_title_obj_part : problem_title_obj_part = "0"
    elif problem_title_obj_part.startswith("+"): problem_title_obj_part = problem_title_obj_part[1:].strip()
    problem_title_str = f"{objective_type.capitalize()} Z = {problem_title_obj_part}"
    logs.append(f"Problem Title: {problem_title_str}")


    # Chuyển đổi constraints từ "Định dạng A" sang định dạng mà _plot_feasible_region mong đợi
    # và logic tìm giao điểm sử dụng ('coefficients', 'type', 'rhs')
    constraints_for_logic_and_plot: List[Dict[str, Any]] = []
    for i, constr_a in enumerate(constraints_input_format_A):
        if not isinstance(constr_a, dict) or not all(k in constr_a for k in ["lhs", "op", "rhs"]):
            logs.append(f"Warning: Skipping malformed constraint at index {i} from Format A: {constr_a}")
            continue
        if len(constr_a["lhs"]) != 2:
            logs.append(f"Warning: Skipping constraint '{constr_a.get('name', f'Unnamed_{i}')}' due to incorrect number of LHS coefficients ({len(constr_a['lhs'])} for 2 variables).")
            continue
        constraints_for_logic_and_plot.append({
            "name": constr_a.get("name", f"Ràng buộc {i+1}"),
            "coefficients": list(constr_a["lhs"]), # Đổi tên từ 'lhs' thành 'coefficients'
            "type": constr_a["op"],             # Đổi tên từ 'op' thành 'type'
            "rhs": constr_a["rhs"]
        })

    if not constraints_for_logic_and_plot:
        logs.append("INFO: No valid constraints provided after parsing. Problem is considered unbounded.")
        plot_base64_no_constr = _plot_feasible_region(problem_title_str, [], [], None, is_unbounded=True, variable_names=variable_names)
        return {"status": "Unbounded", "message": "No valid constraints provided, problem is unbounded.", "plot_image_base64": plot_base64_no_constr}, logs

    try:
        # Tìm các điểm giao cắt của các cặp đường thẳng ràng buộc
        intersection_points = []
        if len(constraints_for_logic_and_plot) >= 2:
            for constr1_idx, constr2_idx in itertools.combinations(range(len(constraints_for_logic_and_plot)), 2):
                constr1 = constraints_for_logic_and_plot[constr1_idx]
                constr2 = constraints_for_logic_and_plot[constr2_idx]
                # Đã kiểm tra len(coefficients) == 2 ở trên
                A = np.array([constr1["coefficients"], constr2["coefficients"]])
                b = np.array([constr1["rhs"], constr2["rhs"]])
                if abs(np.linalg.det(A)) > 1e-9: # Kiểm tra xem 2 đường có cắt nhau không (det != 0)
                    try:
                        point = np.linalg.solve(A, b)
                        if np.all(np.isfinite(point)): # Chỉ thêm điểm hữu hạn
                            intersection_points.append(point)
                            logs.append(f"Intersection of '{constr1.get('name')}' and '{constr2.get('name')}': ({point[0]:.2f}, {point[1]:.2f})")
                    except np.linalg.LinAlgError:
                        logs.append(f"Note: Constraints '{constr1.get('name')}' and '{constr2.get('name')}' are parallel or identical, no unique intersection.")
                        pass # Đường thẳng song song hoặc trùng nhau
        elif len(constraints_for_logic_and_plot) == 1:
            logs.append("INFO: Only one constraint provided. No intersection points from combinations.")

        # Kiểm tra các điểm giao với trục tọa độ cho từng ràng buộc
        # (0, y_intercept) và (x_intercept, 0)
        axis_intercept_points = []
        for constr in constraints_for_logic_and_plot:
            c = constr["coefficients"]
            r = constr["rhs"]
            # Giao với trục x2 (x1=0)
            if abs(c[1]) > 1e-9: # Nếu hệ số của x2 khác 0
                y_intercept = r / c[1]
                axis_intercept_points.append(np.array([0, y_intercept]))
            # Giao với trục x1 (x2=0)
            if abs(c[0]) > 1e-9: # Nếu hệ số của x1 khác 0
                x_intercept = r / c[0]
                axis_intercept_points.append(np.array([x_intercept, 0]))
        
        # Kết hợp các điểm giao và điểm (0,0)
        candidate_points = intersection_points + axis_intercept_points
        # Thêm (0,0) nếu nó chưa có và là một điểm cần xét (ví dụ nếu các ràng buộc phi âm không tường minh)
        # Tuy nhiên, nên dựa vào việc (0,0) có thỏa mãn các ràng buộc không
        # candidate_points.append(np.array([0,0])) # Sẽ được lọc bởi is_feasible

        feasible_vertices = []
        # Lọc các điểm ứng viên để tìm các đỉnh khả thi
        if candidate_points:
            for point in candidate_points:
                x1_p, x2_p = point
                is_feasible_point = True
                for constr_check in constraints_for_logic_and_plot:
                    check_val = constr_check["coefficients"][0] * x1_p + constr_check["coefficients"][1] * x2_p
                    op_check, rhs_check = constr_check["type"], constr_check["rhs"]
                    epsilon_feas = 1e-9 # Ngưỡng cho việc kiểm tra khả thi
                    if not (
                        ((op_check == "<=" or op_check == "≤") and check_val <= rhs_check + epsilon_feas) or \
                        ((op_check == ">=" or op_check == "≥") and check_val >= rhs_check - epsilon_feas) or \
                        ((op_check == "==" or op_check == "=") and abs(check_val - rhs_check) <= epsilon_feas)
                    ):
                        is_feasible_point = False; break
                if is_feasible_point:
                    # Tránh thêm các điểm rất gần nhau
                    is_new_vertex = True
                    for existing_v in feasible_vertices:
                        if np.allclose(point, existing_v, atol=1e-5):
                            is_new_vertex = False; break
                    if is_new_vertex:
                        feasible_vertices.append(point)
                        logs.append(f"Point ({x1_p:.2f}, {x2_p:.2f}) is a feasible vertex.")
        
        # Kiểm tra (0,0) riêng nếu không có điểm nào khác
        if not feasible_vertices :
            is_origin_feasible = True
            for constr_check_origin in constraints_for_logic_and_plot:
                check_val_origin = 0 # Vì x1=0, x2=0
                op_origin, rhs_origin = constr_check_origin["type"], constr_check_origin["rhs"]
                epsilon_feas_origin = 1e-9
                if not (
                    ((op_origin == "<=" or op_origin == "≤") and check_val_origin <= rhs_origin + epsilon_feas_origin) or \
                    ((op_origin == ">=" or op_origin == "≥") and check_val_origin >= rhs_origin - epsilon_feas_origin) or \
                    ((op_origin == "==" or op_origin == "=") and abs(check_val_origin - rhs_origin) <= epsilon_feas_origin)
                ):
                    is_origin_feasible = False; break
            if is_origin_feasible:
                 feasible_vertices.append(np.array([0.0, 0.0]))
                 logs.append(f"Point (0.00, 0.00) is a feasible vertex (checked separately).")


        if not feasible_vertices and len(constraints_for_logic_and_plot) >=1 : # >=1 thay vì >=2
            logs.append("INFO: No feasible vertices found. Problem is likely infeasible or the feasible region is unbounded without vertices.")
            # Vẽ đồ thị để người dùng tự đánh giá
            plot_base64_empty = _plot_feasible_region(problem_title_str, constraints_for_logic_and_plot, [], None, is_unbounded=False, variable_names=variable_names)
            return {"status": "Infeasible", "message": "No feasible vertices found. The feasible region may be empty.", "plot_image_base64": plot_base64_empty}, logs

        # Đánh giá hàm mục tiêu tại các đỉnh khả thi
        objective_coeffs_np = np.array(objective_coeffs_list)
        best_value = None
        optimal_vertex = None
        epsilon_obj_compare = 1e-9

        if feasible_vertices:
            logs.append("\n--- Evaluating objective function at each feasible vertex ---")
            for vertex_idx, vertex_val in enumerate(feasible_vertices):
                current_value = np.dot(objective_coeffs_np, vertex_val)
                logs.append(f"Value at Vertex {chr(65+vertex_idx)} ({vertex_val[0]:.2f}, {vertex_val[1]:.2f}) is Z = {current_value:.2f}")

                if optimal_vertex is None: # Gán lần đầu
                    best_value = current_value
                    optimal_vertex = vertex_val
                else:
                    if objective_type == "maximize":
                        if current_value > best_value + epsilon_obj_compare:
                            best_value = current_value
                            optimal_vertex = vertex_val
                    elif objective_type == "minimize":
                        if current_value < best_value - epsilon_obj_compare:
                            best_value = current_value
                            optimal_vertex = vertex_val
            if optimal_vertex is None and feasible_vertices: # Nếu có đỉnh khả thi nhưng không tìm được tối ưu (hiếm)
                optimal_vertex = feasible_vertices[0]
                best_value = np.dot(objective_coeffs_np, optimal_vertex)
                logs.append(f"Warning: Optimal vertex not decisively found, defaulting to first feasible vertex. Value: {best_value:.2f}")

        # Kiểm tra tính không bị chặn
        is_unbounded_flag = False
        if feasible_vertices or not constraints_for_logic_and_plot: # Nếu có đỉnh hoặc không có ràng buộc nào
            direction_to_check = objective_coeffs_np if objective_type == "maximize" else -objective_coeffs_np
            # Chỉ kiểm tra unbounded nếu direction_to_check khác vector 0
            if np.any(np.abs(direction_to_check) > 1e-9):
                start_point_for_unbounded_check = optimal_vertex if optimal_vertex is not None else np.array([0.0,0.0])
                if not feasible_vertices and constraints_for_logic_and_plot: # Nếu không có đỉnh nhưng có ràng buộc, thử 1 điểm trên 1 ràng buộc
                    # Tìm một điểm bất kỳ thỏa mãn ít nhất một ràng buộc (để bắt đầu kiểm tra unbounded)
                    # Đây là một heuristic đơn giản
                    for constr_probe in constraints_for_logic_and_plot:
                        c_probe = constr_probe["coefficients"]
                        r_probe = constr_probe["rhs"]
                        if abs(c_probe[1]) > 1e-6: start_point_for_unbounded_check = np.array([0, r_probe / c_probe[1]]); break
                        elif abs(c_probe[0]) > 1e-6: start_point_for_unbounded_check = np.array([r_probe / c_probe[0], 0]); break


                large_step_check = 1e5 # Bước lớn để kiểm tra
                test_point_far = start_point_for_unbounded_check + large_step_check * (direction_to_check / (np.linalg.norm(direction_to_check) + 1e-9) ) # Chia để tránh lỗi chia cho 0

                is_far_point_feasible = True
                if not constraints_for_logic_and_plot: # Nếu không có ràng buộc, luôn feasible
                    is_far_point_feasible = True
                else:
                    for constr_unb_check in constraints_for_logic_and_plot:
                        val_at_far_point = constr_unb_check["coefficients"][0] * test_point_far[0] + constr_unb_check["coefficients"][1] * test_point_far[1]
                        op_unb, rhs_unb = constr_unb_check["type"], constr_unb_check["rhs"]
                        epsilon_unb_check = 1.0 # Nới lỏng epsilon cho kiểm tra điểm ở xa
                        if not (
                            ((op_unb == "<=" or op_unb == "≤") and val_at_far_point <= rhs_unb + epsilon_unb_check) or \
                            ((op_unb == ">=" or op_unb == "≥") and val_at_far_point >= rhs_unb - epsilon_unb_check) or \
                            ((op_unb == "==" or op_unb == "=") and abs(val_at_far_point - rhs_unb) <= epsilon_unb_check)
                        ):
                            is_far_point_feasible = False; break
                
                if is_far_point_feasible:
                    value_at_far_point = np.dot(objective_coeffs_np, test_point_far)
                    current_best_for_unbounded_check = best_value if best_value is not None else np.dot(objective_coeffs_np, start_point_for_unbounded_check)
                    
                    if (objective_type == "maximize" and value_at_far_point > current_best_for_unbounded_check + epsilon_obj_compare) or \
                       (objective_type == "minimize" and value_at_far_point < current_best_for_unbounded_check - epsilon_obj_compare):
                        is_unbounded_flag = True
                        logs.append(f"INFO: Problem appears to be UNBOUNDED. Test point far away ({test_point_far[0]:.1f}, {test_point_far[1]:.1f}) is feasible and improves objective.")

        # Vẽ đồ thị
        plot_image_base64_str = _plot_feasible_region(
            problem_title_str,
            constraints_for_logic_and_plot,
            feasible_vertices,
            optimal_vertex if not is_unbounded_flag else None, # Không hiển thị điểm tối ưu nếu không bị chặn
            is_unbounded_flag,
            variable_names
        )

        if is_unbounded_flag:
            return {"status": "Unbounded", "message": "The objective function can be improved indefinitely.", "plot_image_base64": plot_image_base64_str}, logs

        if optimal_vertex is not None and best_value is not None:
            solution_dict = {
                "status": "Optimal",
                "objective_value": best_value,
                "variables": {
                    variable_names[0]: optimal_vertex[0],
                    variable_names[1]: optimal_vertex[1]
                },
                "plot_image_base64": plot_image_base64_str
            }
            logs.append(f"\nOptimal solution found at ({optimal_vertex[0]:.2f}, {optimal_vertex[1]:.2f}) with objective value {best_value:.2f}")
            return solution_dict, logs
        else: # Không tìm thấy đỉnh tối ưu mặc dù có thể có đỉnh khả thi (hoặc không có đỉnh nào)
            logs.append("INFO: Could not find an optimal solution. The feasible region might be empty, or problem structure is unusual.")
            status_msg = "InfeasibleOrSpecialCase"
            # Nếu không có đỉnh khả thi nào được tìm thấy VÀ có ít nhất một ràng buộc, thì có thể là Infeasible.
            if not feasible_vertices and constraints_for_logic_and_plot:
                status_msg = "Infeasible"
            return {"status": status_msg, "message": "No optimal solution determined by geometric method.", "plot_image_base64": plot_image_base64_str}, logs

    except Exception as e:
        error_msg_exc = f"An unexpected error occurred in geometric solver: {e}"
        logs.append(f"ERROR: {error_msg_exc}")
        logger.exception(error_msg_exc)
        try: # Cố gắng vẽ plot lỗi
            plot_base64_fallback_exc = _plot_feasible_region(problem_title_str if 'problem_title_str' in locals() else "Lỗi Vẽ Đồ Thị", constraints_for_logic_and_plot if 'constraints_for_logic_and_plot' in locals() else [], [], None, is_unbounded=True, variable_names=variable_names if 'variable_names' in locals() else ["x1","x2"])
        except:
            plot_base64_fallback_exc = None
        return {"status": "Error", "message": error_msg_exc, "plot_image_base64": plot_base64_fallback_exc}, logs

if __name__ == '__main__':
    import base64
    import os
    import webbrowser

    logging.basicConfig(level=logging.INFO, format='%(levelname)s (geom_solver): %(message)s')

    # Ví dụ sử dụng "Định dạng A"
    problem_A_optimal = {
        "objective": "maximize",
        "coeffs": [3, 2], # Max 3x1 + 2x2
        "variables_names_for_title_only": ["X_mot", "Y_hai"],
        "constraints": [
            {"name": "R1", "lhs": [1, 1], "op": "<=", "rhs": 4},
            {"name": "R2", "lhs": [2, 1], "op": "<=", "rhs": 5},
            {"name": "X_nonneg", "lhs": [1,0], "op": ">=", "rhs": 0}, # X_mot >= 0
            {"name": "Y_nonneg", "lhs": [0,1], "op": ">=", "rhs": 0}  # Y_hai >= 0
        ]
        # Lời giải x1=1, x2=3, Z=9
    }
    print("--- Test 1: Optimal Problem (Format A) ---")
    solution1, logs1 = solve_with_geometric_method(problem_A_optimal)
    # for log_entry in logs1: print(log_entry)
    if solution1:
        print(f"Status: {solution1.get('status')}, Objective: {solution1.get('objective_value')}, Vars: {solution1.get('variables')}")
        if solution1.get("plot_image_base64"):
            try:
                header, encoded = solution1["plot_image_base64"].split(",",1)
                img_data = base64.b64decode(encoded)
                path = "test_outputs/geom_test1_optimal.png"; os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f: f.write(img_data)
                print(f"Saved plot to {path}")
                # webbrowser.open(os.path.realpath(path))
            except Exception as e_img: print(f"Error saving/opening image: {e_img}")


    problem_A_unbounded = {
        "objective": "maximize",
        "coeffs": [1, 1],
        "variables_names_for_title_only": ["u", "v"],
        "constraints": [
            {"name": "R1", "lhs": [1, -1], "op": "<=", "rhs": 1}, # u - v <= 1
            {"name": "u_nonneg", "lhs": [1,0], "op": ">=", "rhs": 0},
            {"name": "v_nonneg", "lhs": [0,1], "op": ">=", "rhs": 0}
        ]
    }
    print("\n--- Test 2: Unbounded Problem (Format A) ---")
    solution2, logs2 = solve_with_geometric_method(problem_A_unbounded)
    # for log_entry in logs2: print(log_entry)
    if solution2:
        print(f"Status: {solution2.get('status')}")
        if solution2.get("plot_image_base64"):
            try:
                header, encoded = solution2["plot_image_base64"].split(",",1)
                img_data = base64.b64decode(encoded)
                path = "test_outputs/geom_test2_unbounded.png"; os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f: f.write(img_data)
                print(f"Saved plot to {path}")
            except Exception as e_img: print(f"Error saving image: {e_img}")


    problem_A_infeasible = {
        "objective": "maximize",
        "coeffs": [1, 1],
        "variables_names_for_title_only": ["a", "b"],
        "constraints": [
            {"name": "R1", "lhs": [1, 0], "op": "<=", "rhs": 1}, # a <= 1
            {"name": "R2", "lhs": [1, 0], "op": ">=", "rhs": 2}  # a >= 2
        ]
    }
    print("\n--- Test 3: Infeasible Problem (Format A) ---")
    solution3, logs3 = solve_with_geometric_method(problem_A_infeasible)
    # for log_entry in logs3: print(log_entry)
    if solution3:
        print(f"Status: {solution3.get('status')}")
        if solution3.get("plot_image_base64"):
            try:
                header, encoded = solution3["plot_image_base64"].split(",",1)
                img_data = base64.b64decode(encoded)
                path = "test_outputs/geom_test3_infeasible.png"; os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f: f.write(img_data)
                print(f"Saved plot to {path}")
            except Exception as e_img: print(f"Error saving image: {e_img}")

    problem_A_3vars = {
        "objective": "maximize",
        "coeffs": [1,1,1],
        "variables_names_for_title_only": ["x","y","z"],
        "constraints":[]
    }
    print("\n--- Test 4: 3 Variables Error (Format A) ---")
    solution4, logs4 = solve_with_geometric_method(problem_A_3vars)
    if solution4: print(f"Status: {solution4.get('status')}, Message: {solution4.get('message')}")


    problem_A_no_constraints = {
        "objective": "maximize",
        "coeffs": [1,1],
        "variables_names_for_title_only": ["p","q"],
        "constraints":[]
    }
    print("\n--- Test 5: No Constraints (Format A) ---")
    solution5, logs5 = solve_with_geometric_method(problem_A_no_constraints)
    if solution5: print(f"Status: {solution5.get('status')}, Message: {solution5.get('message')}")

