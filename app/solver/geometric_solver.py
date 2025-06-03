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

def _normalize_problem_data(problem_data_input: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Chuẩn hóa problem_data từ định dạng NLP mới sang định dạng mà solver mong đợi.
    """
    logs = []
    if "coeffs" in problem_data_input and isinstance(problem_data_input.get("objective"), str):
        logs.append("Normalizing input data from new NLP format for geometric solver.")
        normalized_data: Dict[str, Any] = {"objective": {}, "variables": [], "constraints": []}
        
        obj_type_str = problem_data_input["objective"].lower()
        if obj_type_str == "max":
            normalized_data["objective"]["type"] = "maximize"
        elif obj_type_str == "min":
            normalized_data["objective"]["type"] = "minimize"
        else:
            logs.append(f"ERROR: Invalid objective type '{obj_type_str}' in new format.")
            return None, logs
            
        obj_coeffs = problem_data_input.get("coeffs", [])
        if not isinstance(obj_coeffs, list):
            logs.append("ERROR: 'coeffs' for objective must be a list in new format.")
            return None, logs
        normalized_data["objective"]["coefficients"] = obj_coeffs
        
        num_vars = len(obj_coeffs)
        if num_vars == 0 and problem_data_input.get("constraints"):
            first_constr = problem_data_input["constraints"][0]
            if isinstance(first_constr, dict) and "lhs" in first_constr and isinstance(first_constr["lhs"], list):
                num_vars = len(first_constr["lhs"])
        
        if num_vars == 0: 
            logs.append("ERROR: Cannot determine number of variables from new format.")
            return None, logs

        # Sử dụng tên biến từ "variables_names_for_title_only" nếu có, nếu không thì tự tạo x1, x2...
        # Đảm bảo danh sách này có đủ tên cho num_vars
        provided_var_names = problem_data_input.get("variables_names_for_title_only", [])
        final_var_names = [f"x{i+1}" for i in range(num_vars)]
        for i in range(min(num_vars, len(provided_var_names))):
            final_var_names[i] = provided_var_names[i]
        normalized_data["variables"] = final_var_names
        
        raw_constraints = problem_data_input.get("constraints", [])
        if not isinstance(raw_constraints, list):
            logs.append("ERROR: 'constraints' must be a list in new format.")
            return None, logs

        for i, constr_new in enumerate(raw_constraints):
            if not isinstance(constr_new, dict):
                logs.append(f"ERROR: Constraint at index {i} is not a dictionary.")
                return None, logs

            lhs = constr_new.get("lhs")
            op = constr_new.get("op")
            rhs = constr_new.get("rhs")

            if not all([isinstance(lhs, list), isinstance(op, str), isinstance(rhs, (int, float))]):
                logs.append(f"ERROR: Invalid structure or types in constraint {i+1} (lhs, op, rhs).")
                return None, logs
            
            if len(lhs) != num_vars:
                logs.append(f"ERROR: Number of coefficients in LHS of constraint {i+1} ({len(lhs)}) does not match number of variables ({num_vars}).")
                return None, logs

            normalized_data["constraints"].append({
                "name": constr_new.get("name", f"({i+1})"), 
                "coefficients": lhs,
                "type": op,
                "rhs": rhs
            })
        logs.append("Normalization successful for geometric solver.")
        return normalized_data, logs
    
    logs.append("Input data appears to be in the expected (old) format for geometric solver or is invalid.")
    if not (isinstance(problem_data_input.get("objective"), dict) and \
            "type" in problem_data_input["objective"] and \
            "coefficients" in problem_data_input["objective"] and \
            isinstance(problem_data_input.get("variables"), list) and \
            isinstance(problem_data_input.get("constraints"), list)):
        logs.append("ERROR: Old format data is also invalid.")
        return None, logs
        
    return problem_data_input, logs


def _plot_feasible_region(problem_title:str, constraints_to_draw: List[Dict], feasible_vertices: List[np.ndarray], optimal_vertex: Optional[np.ndarray] = None, is_unbounded: bool = False) -> str:
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
        sorted_vertices = sorted(feasible_vertices, key=lambda v: np.arctan2(v[1] - center[1], v[0] - center[0]))
        polygon = plt.Polygon(sorted_vertices, color='lightcyan', alpha=0.7, ec='teal', linewidth=1.5) 
        ax.add_patch(polygon)
    elif not is_unbounded and feasible_vertices and len(feasible_vertices) == 2: 
        ax.plot([v[0] for v in feasible_vertices], [v[1] for v in feasible_vertices], color='teal', linewidth=3, alpha=0.7)
    elif not is_unbounded and feasible_vertices and len(feasible_vertices) == 1: 
         ax.plot(feasible_vertices[0][0], feasible_vertices[0][1], 'o', color='teal', markersize=8, alpha=0.7)

    padding_factor = 0.3 
    x_coords_for_lim = [0.0] 
    y_coords_for_lim = [0.0]
    
    if feasible_vertices:
        x_coords_for_lim.extend([v[0] for v in feasible_vertices])
        y_coords_for_lim.extend([v[1] for v in feasible_vertices])
    if optimal_vertex is not None and not is_unbounded: 
        x_coords_for_lim.append(optimal_vertex[0])
        y_coords_for_lim.append(optimal_vertex[1])

    # Tính toán giới hạn dựa trên các điểm đã biết hoặc các giao điểm của ràng buộc với trục
    if not x_coords_for_lim or len(x_coords_for_lim) <= 1: x_coords_for_lim.extend([-1.0, 1.0]) # Mặc định
    if not y_coords_for_lim or len(y_coords_for_lim) <= 1: y_coords_for_lim.extend([-1.0, 1.0])

    for constr in constraints_to_draw: 
        c = constr["coefficients"]
        r = constr["rhs"]
        if abs(c[1]) > 1e-6: # Giao với trục x (y=0)
            if abs(c[0]) > 1e-6 : x_coords_for_lim.append(r/c[0]) 
        if abs(c[0]) > 1e-6: # Giao với trục y (x=0)
            if abs(c[1]) > 1e-6: y_coords_for_lim.append(r/c[1])

    x_min_lim, x_max_lim = min(x_coords_for_lim), max(x_coords_for_lim)
    y_min_lim, y_max_lim = min(y_coords_for_lim), max(y_coords_for_lim)
    
    x_current_range = (x_max_lim - x_min_lim) if (x_max_lim - x_min_lim) > 1e-1 else 4.0 
    y_current_range = (y_max_lim - y_min_lim) if (y_max_lim - y_min_lim) > 1e-1 else 4.0
    
    ax.set_xlim(x_min_lim - x_current_range * padding_factor - 0.5, x_max_lim + x_current_range * padding_factor + 0.5) 
    ax.set_ylim(y_min_lim - y_current_range * padding_factor - 0.5, y_max_lim + y_current_range * padding_factor + 0.5)

    line_x_coords_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200) 
    
    cmap = plt.cm.get_cmap('tab10') 
    num_total_colors = cmap.N if hasattr(cmap, 'N') else 10 

    for i, constr in enumerate(constraints_to_draw): 
        c = constr["coefficients"]
        r = constr["rhs"]
        op = constr["type"]
        constr_name = constr.get("name", f"({i+1})") 
        current_line_color = cmap(i % num_total_colors) 

        line_drawn = False
        # Chọn một điểm trên đường thẳng để đặt mũi tên và nhãn
        # Ưu tiên điểm giữa của phần nhìn thấy của đường thẳng
        mid_point_for_arrow = None

        if abs(c[1]) > 1e-6: # Đường không thẳng đứng
            y_vals_line = (r - c[0] * line_x_coords_plot) / c[1]
            line, = ax.plot(line_x_coords_plot, y_vals_line, label=constr_name, alpha=0.9, linestyle='-', color=current_line_color, linewidth=1.5)
            line_drawn = True
            # Tìm điểm giữa của đoạn thẳng hiển thị trên plot
            visible_indices = np.where((y_vals_line >= ax.get_ylim()[0]) & (y_vals_line <= ax.get_ylim()[1]))[0]
            if len(visible_indices) > 0:
                mid_idx = visible_indices[len(visible_indices) // 2]
                mid_point_for_arrow = (line_x_coords_plot[mid_idx], y_vals_line[mid_idx])

        elif abs(c[0]) > 1e-6: # Đường thẳng đứng
            x_val_line = r / c[0]
            y_coords_for_vertical_line = np.array([ax.get_ylim()[0], ax.get_ylim()[1]]) # Vẽ từ biên dưới lên biên trên
            line, = ax.plot([x_val_line, x_val_line], y_coords_for_vertical_line, label=constr_name, alpha=0.9, linestyle='-', color=current_line_color, linewidth=1.5)
            line_drawn = True
            mid_point_for_arrow = (x_val_line, np.mean(y_coords_for_vertical_line)) # Điểm giữa đường thẳng đứng
        
        if line_drawn and mid_point_for_arrow: 
            arrow_color_to_use = line.get_color() 
            normal_vector = np.array([c[0], c[1]]) 
            norm = np.linalg.norm(normal_vector)
            if norm > 1e-6 :
                normal_vector_unit = normal_vector / norm 
            else:
                continue 
            
            plot_diag_len = np.sqrt((ax.get_xlim()[1]-ax.get_xlim()[0])**2 + (ax.get_ylim()[1]-ax.get_ylim()[0])**2)
            actual_arrow_length = plot_diag_len / 25.0 # Tăng độ dài mũi tên
            
            direction_sign = 0 
            if op == "<=" or op == "≤": direction_sign = -1 
            elif op == ">=" or op == "≥": direction_sign = 1 
            
            if direction_sign != 0:
                mid_x, mid_y = mid_point_for_arrow
                arrow_dir_x_comp = direction_sign * normal_vector_unit[0] 
                arrow_dir_y_comp = direction_sign * normal_vector_unit[1]
                
                # Điểm bắt đầu của mũi tên: từ điểm trên đường thẳng, dịch ra một chút theo hướng mũi tên
                base_offset_len = actual_arrow_length * 0.2 
                arrow_base_x = mid_x + arrow_dir_x_comp * base_offset_len
                arrow_base_y = mid_y + arrow_dir_y_comp * base_offset_len
                
                # Điểm kết thúc của mũi tên
                arrow_tip_x = arrow_base_x + arrow_dir_x_comp * actual_arrow_length
                arrow_tip_y = arrow_base_y + arrow_dir_y_comp * actual_arrow_length

                ax.annotate("", xy=(arrow_tip_x, arrow_tip_y), 
                            xytext=(arrow_base_x, arrow_base_y),
                            arrowprops=dict(arrowstyle="-|>", color=arrow_color_to_use, lw=1.2, mutation_scale=15))
                
                # Đặt text gần điểm giữa của mũi tên (từ base đến tip)
                text_pos_x = (arrow_base_x + arrow_tip_x) / 2
                text_pos_y = (arrow_base_y + arrow_tip_y) / 2
                # Dịch text ra ngoài một chút so với thân mũi tên
                text_offset_from_arrow_body = actual_arrow_length * 0.3
                text_x = text_pos_x + arrow_dir_x_comp * text_offset_from_arrow_body
                text_y = text_pos_y + arrow_dir_y_comp * text_offset_from_arrow_body
                
                ax.text(text_x, text_y, constr_name, fontsize=7, color=arrow_color_to_use, 
                        ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.15', fc='white', alpha=0.7, ec='none'))
    
    # ... (phần còn lại của hàm _plot_feasible_region: vẽ đỉnh, điểm tối ưu, trục, grid, legend) ...
    current_x_range_plot = ax.get_xlim()[1] - ax.get_xlim()[0]
    current_y_range_plot = ax.get_ylim()[1] - ax.get_ylim()[0]

    if feasible_vertices:
        for i, v_point in enumerate(feasible_vertices):
            ax.plot(v_point[0], v_point[1], 'o', color='crimson', markersize=6, label='Đỉnh Khả Thi' if i==0 and not is_unbounded else None) 
            ax.text(v_point[0] + 0.02 * current_x_range_plot, v_point[1] + 0.02 * current_y_range_plot, f'{chr(65+i)} ({v_point[0]:.2f}, {v_point[1]:.2f})', fontsize=8)

    if optimal_vertex is not None and not is_unbounded:
        ax.plot(optimal_vertex[0], optimal_vertex[1], 'o', color='forestgreen', markersize=10, mec='black', label='Điểm Tối Ư')
        ax.text(optimal_vertex[0] + 0.02 * current_x_range_plot, optimal_vertex[1] - 0.04 * current_y_range_plot, f'Tối ưu\n({optimal_vertex[0]:.2f}, {optimal_vertex[1]:.2f})', fontsize=9, color='forestgreen', fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='lightgray')
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles: 
        unique_labels_dict = {} 
        for handle, label_text in zip(handles, labels): 
            if label_text not in unique_labels_dict:
                unique_labels_dict[label_text] = handle
        ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(), fontsize=9, loc='best')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120) 
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    
    return f"data:image/png;base64,{img_base64}"


def solve_with_geometric_method(problem_data_input: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    overall_logs = []
    
    # Bước 0: Chuẩn hóa dữ liệu đầu vào
    problem_data, norm_logs = _normalize_problem_data(problem_data_input)
    overall_logs.extend(norm_logs)

    if problem_data is None:
        logger.error("Failed to normalize problem data for geometric solver.")
        fallback_title = "Lỗi xử lý bài toán"
        if isinstance(problem_data_input, dict) and "objective" in problem_data_input:
            obj_val = problem_data_input.get('objective','Unknown objective')
            coeffs_val = problem_data_input.get('coeffs', [])
            coeffs_str = ", ".join(map(str, coeffs_val))
            fallback_title = f"{str(obj_val).capitalize()} z = {coeffs_str}..." if coeffs_str else f"{str(obj_val).capitalize()} z = ..."
        plot_base64_error_norm = _plot_feasible_region(fallback_title, [], [], None, is_unbounded=True) 
        return {"status": "Error", "message": "Invalid input data format after normalization attempt.", "plot_image_base64": plot_base64_error_norm}, overall_logs

    logs = ["--- Starting Geometric Solver with (potentially normalized) data ---"] 
    
    obj_expr_terms = []
    # Xây dựng tiêu đề bài toán từ problem_data (đã được chuẩn hóa)
    if "objective" in problem_data and "coefficients" in problem_data["objective"] and "variables" in problem_data:
        for i, coeff_val in enumerate(problem_data['objective']['coefficients']):
            var_name = problem_data['variables'][i]
            term_str = ""
            # Xử lý hệ số 1 và -1 cho đẹp
            if abs(coeff_val) == 1: # Nếu hệ số là 1 hoặc -1
                 term_str = f"{'-' if coeff_val < 0 else ''}{var_name}"
            elif coeff_val != 0:
                 term_str = f"{coeff_val}{var_name}"
            
            if term_str: 
                if obj_expr_terms: # Nếu không phải số hạng đầu tiên
                    if coeff_val > 0:
                        obj_expr_terms.append(f"+ {term_str}")
                    else: # coeff_val < 0 (dấu - đã có trong term_str)
                         obj_expr_terms.append(f" {term_str}") 
                else: # Số hạng đầu tiên
                    obj_expr_terms.append(term_str)
        
        problem_title_obj_part = "".join(obj_expr_terms).strip()
        if problem_title_obj_part.startswith("+"): problem_title_obj_part = problem_title_obj_part[1:].strip()
        if not problem_title_obj_part: problem_title_obj_part = "0" 
        problem_title = f"{problem_data['objective']['type'].capitalize()} z = {problem_title_obj_part}"
    else:
        problem_title = "Bài toán Quy hoạch Tuyến tính"


    if len(problem_data.get("variables", [])) != 2:
        error_msg = "Geometric method only supports problems with exactly 2 variables."
        logs.append(f"ERROR: {error_msg}")
        logger.error(error_msg)
        overall_logs.extend(logs)
        # Trả về plot rỗng với tiêu đề lỗi
        plot_base64_error_vars = _plot_feasible_region(problem_title + " (Lỗi số biến)", [], [], None, is_unbounded=True)
        return {"status": "Error", "message": error_msg, "plot_image_base64": plot_base64_error_vars}, overall_logs

    try:
        # CHỈ SỬ DỤNG RÀNG BUỘC TỪ INPUT ĐÃ ĐƯỢC CHUẨN HÓA
        all_constraints_for_logic = problem_data.get("constraints", [])
        
        if not all_constraints_for_logic: 
            logs.append("INFO: No constraints provided. Problem is considered unbounded.")
            plot_base64_no_constraints = _plot_feasible_region(problem_title, [], [], None, is_unbounded=True) 
            overall_logs.extend(logs)
            return {"status": "Unbounded", "message": "No constraints provided, problem is unbounded.", "plot_image_base64": plot_base64_no_constraints}, overall_logs

        intersection_points = []
        if len(all_constraints_for_logic) >= 2:
            for constr1_idx, constr2_idx in itertools.combinations(range(len(all_constraints_for_logic)), 2):
                constr1 = all_constraints_for_logic[constr1_idx]
                constr2 = all_constraints_for_logic[constr2_idx]
                # Đảm bảo coefficients là list có 2 phần tử
                if len(constr1.get("coefficients",[])) != 2 or len(constr2.get("coefficients",[])) != 2:
                    logs.append(f"Warning: Skipping intersection due to incorrect coefficient count in constraints '{constr1.get('name',f'({constr1_idx+1})')}' or '{constr2.get('name',f'({constr2_idx+1})')}'.")
                    continue
                A = np.array([constr1["coefficients"], constr2["coefficients"]])
                b = np.array([constr1["rhs"], constr2["rhs"]])
                if abs(np.linalg.det(A)) > 1e-9: 
                    try:
                        point = np.linalg.solve(A, b)
                        if np.all(np.isfinite(point)):
                            intersection_points.append(point)
                            logs.append(f"Intersection of '{constr1.get('name',f'({constr1_idx+1})')}' and '{constr2.get('name',f'({constr2_idx+1})')}': ({point[0]:.2f}, {point[1]:.2f})")
                    except np.linalg.LinAlgError: pass
        elif len(all_constraints_for_logic) == 1:
            logs.append("INFO: Only one constraint provided. No intersection points from combinations. Feasible region is a half-plane.")
        
        feasible_vertices = []
        if intersection_points: 
            for point in intersection_points:
                x1_p, x2_p = point 
                is_feasible = True
                for constr_item_check in all_constraints_for_logic: 
                    # Đảm bảo coefficients là list có 2 phần tử
                    if len(constr_item_check.get("coefficients",[])) != 2:
                        logs.append(f"Warning: Skipping feasibility check for constraint '{constr_item_check.get('name')}' due to incorrect coefficient count.")
                        is_feasible = False; break # Coi như không khả thi nếu ràng buộc lỗi
                    
                    check_val = constr_item_check["coefficients"][0] * x1_p + constr_item_check["coefficients"][1] * x2_p
                    op_check, rhs_check = constr_item_check["type"], constr_item_check["rhs"] 
                    epsilon = 1e-9
                    if not (
                        ((op_check == "<=" or op_check == "≤") and check_val <= rhs_check + epsilon) or \
                        ((op_check == ">=" or op_check == "≥") and check_val >= rhs_check - epsilon) or \
                        ((op_check == "==" or op_check == "=") and abs(check_val - rhs_check) <= epsilon)
                    ):
                        is_feasible = False; break
                if is_feasible:
                    if not any(np.allclose(point, v_exist, atol=1e-5) for v_exist in feasible_vertices):
                        feasible_vertices.append(point)
                        logs.append(f"Point ({x1_p:.2f}, {x2_p:.2f}) is a feasible vertex.")
        
        if not feasible_vertices and len(all_constraints_for_logic) >= 2 :
            logs.append("INFO: No feasible vertices found from intersections. Problem is likely infeasible.")
            plot_base64_empty = _plot_feasible_region(problem_title, all_constraints_for_logic, [], None, is_unbounded=False) 
            overall_logs.extend(logs)
            return {"status": "Infeasible", "message": "No feasible vertices found from intersections.", "plot_image_base64": plot_base64_empty}, overall_logs

        objective_coeffs = np.array(problem_data["objective"]["coefficients"])
        objective_type = problem_data["objective"]["type"]
        best_value = None
        optimal_vertex = None
        epsilon_obj = 1e-9 
        
        if feasible_vertices:
            logs.append("\n--- Evaluating objective function at each vertex ---")
            for vertex in feasible_vertices: 
                current_value = np.dot(objective_coeffs, vertex)
                logs.append(f"Value at ({vertex[0]:.2f}, {vertex[1]:.2f}) is {current_value:.2f}")

                if optimal_vertex is None:
                    best_value = current_value
                    optimal_vertex = vertex
                else:
                    if objective_type == "maximize":
                        if current_value > best_value + epsilon_obj: 
                            best_value = current_value
                            optimal_vertex = vertex
                    elif objective_type == "minimize":
                        if current_value < best_value - epsilon_obj: 
                            best_value = current_value
                            optimal_vertex = vertex
        
            if optimal_vertex is None : # Điều này không nên xảy ra nếu feasible_vertices không rỗng
                 logs.append("ERROR: Could not determine optimal solution despite having feasible vertices (internal logic error).")
                 plot_base64_error = _plot_feasible_region(problem_title, all_constraints_for_logic, feasible_vertices, None)
                 overall_logs.extend(logs)
                 return {"status": "Error", "message": "Could not determine optimal solution from feasible vertices.", "plot_image_base64": plot_base64_error}, overall_logs
        
        is_unbounded = False
        check_start_point = None
        if optimal_vertex is not None:
            check_start_point = optimal_vertex
        elif len(all_constraints_for_logic) == 1 and not feasible_vertices: 
            c0_single, c1_single = all_constraints_for_logic[0]["coefficients"]
            r0_single = all_constraints_for_logic[0]["rhs"]
            if abs(c1_single) > 1e-6: check_start_point = np.array([0.0, r0_single/c1_single])
            elif abs(c0_single) > 1e-6: check_start_point = np.array([r0_single/c0_single, 0.0])
            else: check_start_point = np.array([0.0,0.0]) 
            logs.append(f"INFO: No vertices from intersections, checking for unboundedness from a point on the single constraint boundary: {check_start_point}")

        if check_start_point is not None: 
            grad_direction = objective_coeffs if objective_type == "maximize" else -objective_coeffs
            norm_grad = np.linalg.norm(grad_direction)
            if norm_grad > 1e-9:
                grad_direction_unit = grad_direction / norm_grad
                
                large_step = 1e4 
                test_point_unbounded = check_start_point + large_step * grad_direction_unit
                
                x1_tp, x2_tp = test_point_unbounded
                point_is_feasible_far_away = True
                if not all_constraints_for_logic: point_is_feasible_far_away = True 
                else:
                    for constr_item_unbounded_check in all_constraints_for_logic: 
                        check_val_unbounded = constr_item_unbounded_check["coefficients"][0] * x1_tp + constr_item_unbounded_check["coefficients"][1] * x2_tp
                        op_unbounded, rhs_unbounded = constr_item_unbounded_check["type"], constr_item_unbounded_check["rhs"] 
                        unbounded_check_epsilon = 1.0 
                        if not (
                            ((op_unbounded == "<=" or op_unbounded == "≤") and check_val_unbounded <= rhs_unbounded + unbounded_check_epsilon) or \
                            ((op_unbounded == ">=" or op_unbounded == "≥") and check_val_unbounded >= rhs_unbounded - unbounded_check_epsilon) or \
                            ((op_unbounded == "==" or op_unbounded == "=") and abs(check_val_unbounded - rhs_unbounded) <= unbounded_check_epsilon) 
                        ):
                            point_is_feasible_far_away = False; break
                
                if point_is_feasible_far_away:
                    test_value_unbounded = np.dot(objective_coeffs, test_point_unbounded)
                    current_best_to_compare = best_value if best_value is not None else np.dot(objective_coeffs, check_start_point)

                    if (objective_type == "maximize" and test_value_unbounded > current_best_to_compare + epsilon_obj) or \
                       (objective_type == "minimize" and test_value_unbounded < current_best_to_compare - epsilon_obj):
                        is_unbounded = True
                        logs.append(f"INFO: Problem appears to be UNBOUNDED. Test point ({test_point_unbounded[0]:.2f}, {test_point_unbounded[1]:.2f}) from start ({check_start_point[0]:.2f}, {check_start_point[1]:.2f}) is feasible and objective can be improved.")
        
        plot_base64 = _plot_feasible_region(problem_title, all_constraints_for_logic, feasible_vertices, optimal_vertex if not is_unbounded else None, is_unbounded)
        
        if is_unbounded:
            overall_logs.extend(logs)
            return {"status": "Unbounded", "message": "The objective function can be improved indefinitely.", "plot_image_base64": plot_base64}, overall_logs
        
        if optimal_vertex is not None and best_value is not None: 
            variable_names = problem_data["variables"]
            solution = {
                "status": "Optimal",
                "objective_value": best_value,
                "variables": {
                    variable_names[0]: optimal_vertex[0],
                    variable_names[1]: optimal_vertex[1]
                },
                "plot_image_base64": plot_base64
            }
            logs.append(f"\nOptimal solution found at ({optimal_vertex[0]:.2f}, {optimal_vertex[1]:.2f}) with objective value {best_value:.2f}")
            overall_logs.extend(logs)
            return solution, overall_logs
        else: 
            logs.append("INFO: Could not find an optimal solution. The feasible region might be empty or the problem structure is unusual (e.g., single constraint not leading to unboundedness in obj direction).")
            overall_logs.extend(logs)
            final_status = "InfeasibleOrSpecialCase"
            if not feasible_vertices and not is_unbounded and len(all_constraints_for_logic) >=2 : final_status = "Infeasible" # Chỉ Infeasible nếu có >=2 ràng buộc mà không có đỉnh

            return {"status": final_status, "message": "No optimal solution found. Check logs for details.", "plot_image_base64": plot_base64}, overall_logs

    except Exception as e:
        error_msg = f"An unexpected error occurred in geometric solver: {e}"
        logs.append(f"ERROR: {error_msg}")
        logger.exception(error_msg)
        try:
            plot_base64_fallback = _plot_feasible_region(problem_title, problem_data.get("constraints", []), [], None, is_unbounded=True) 
        except:
            plot_base64_fallback = None 
        overall_logs.extend(logs)
        return {"status": "Error", "message": error_msg, "plot_image_base64": plot_base64_fallback}, overall_logs

if __name__ == '__main__':
    import base64
    import os
    import webbrowser 

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') 
    
    # Ví dụ 1: Sử dụng định dạng NLP MỚI cho bài toán không bị chặn
    # Các ràng buộc x1 >= 0, x2 >= 0 phải được cung cấp tường minh
    new_format_problem_unbounded = {
        "objective": "max", 
        "coeffs": [1, 1],   
        "constraints": [    
            {"name": "c1: x1-x2 <= 1", "lhs": [1, -1], "op": "<=", "rhs": 1},
            {"name": "x1 >= 0", "lhs": [1,0], "op": ">=", "rhs": 0}, 
            {"name": "x2 >= 0", "lhs": [0,1], "op": ">=", "rhs": 0}  
        ]
    }
    
    print("--- BẮT ĐẦU GIẢI BẰNG PHƯƠNG PHÁP HÌNH HỌC (BÀI TOÁN UNBOUNDED - ĐỊNH DẠNG MỚI) ---")
    solution_unb, logs_unb = solve_with_geometric_method(new_format_problem_unbounded)
    
    print("\n--- LOG CHI TIẾT (UNBOUNDED) ---\n")
    for log_entry in logs_unb: print(log_entry)
    print("\n--- KẾT QUẢ CUỐI CÙNG (UNBOUNDED) ---\n")
    if solution_unb:
        plot_image_data_unb = solution_unb.pop("plot_image_base64", None)
        import json
        print(json.dumps(solution_unb, indent=2))
        if plot_image_data_unb:
            print("\n--- XỬ LÝ HÌNH ẢNH (UNBOUNDED) ---")
            try:
                output_dir = "test_outputs"; os.makedirs(output_dir, exist_ok=True)
                header, encoded = plot_image_data_unb.split(",", 1)
                image_data = base64.b64decode(encoded)
                output_path = os.path.join(output_dir, "geometric_solver_unbounded_new_format.png")
                with open(output_path, "wb") as f: f.write(image_data)
                print(f"[INFO] Đã lưu hình ảnh đồ thị vào: {output_path}")
                webbrowser.open(os.path.realpath(output_path))
            except Exception as e: print(f"[ERROR] Không thể xử lý ảnh: {e}")


    # Ví dụ 2: Bài toán từ hình ảnh sách giáo khoa (ĐỊNH DẠNG MỚI)
    # min z = -x1 + x2
    # (1) -x1 - 2x2 <= 6
    # (2)  x1 - 2x2 <= 4
    # (3) -x1 +  x2 <= 1
    # (4)  x1       <= 0
    # (5)        x2 <= 0
    new_format_problem_from_image = {
        "objective": "min", 
        "coeffs": [-1, 1],
        "constraints": [
            {"name":"(1)", "lhs": [-1, -2], "op": "<=", "rhs": 6}, 
            {"name":"(2)", "lhs": [1, -2], "op": "<=", "rhs": 4},  
            {"name":"(3)", "lhs": [-1, 1], "op": "<=", "rhs": 1},  
            {"name":"(4) x1<=0", "lhs": [1, 0], "op": "<=", "rhs": 0},   
            {"name":"(5) x2<=0", "lhs": [0, 1], "op": "<=", "rhs": 0},   
        ]
    }
    print("\n\n--- BẮT ĐẦU GIẢI BẰNG PHƯƠNG PHÁP HÌNH HỌC (BÀI TOÁN TỪ HÌNH ẢNH - ĐỊNH DẠNG MỚI) ---")
    solution_img, logs_img = solve_with_geometric_method(new_format_problem_from_image)
    print("\n--- LOG CHI TIẾT (IMAGE PROBLEM) ---\n")
    for log_entry in logs_img: print(log_entry)
    print("\n--- KẾT QUẢ CUỐI CÙNG (IMAGE PROBLEM) ---\n")
    if solution_img:
        plot_image_data_img = solution_img.pop("plot_image_base64", None)
        print(json.dumps(solution_img, indent=2))
        if plot_image_data_img:
            print("\n--- XỬ LÝ HÌNH ẢNH (IMAGE PROBLEM) ---")
            try:
                output_dir = "test_outputs"; os.makedirs(output_dir, exist_ok=True)
                header, encoded = plot_image_data_img.split(",", 1)
                image_data = base64.b64decode(encoded)
                output_path = os.path.join(output_dir, "geometric_solver_textbook_new_format.png") 
                with open(output_path, "wb") as f: f.write(image_data)
                print(f"[INFO] Đã lưu hình ảnh đồ thị vào: {output_path}")
                webbrowser.open(os.path.realpath(output_path))
            except Exception as e: print(f"[ERROR] Không thể xử lý ảnh: {e}")
    
    # Ví dụ 3: Bài toán từ hình vẽ tay của bạn (max 3x1+2x2, x1+2x2<=6, -x1+x2<=1, x1>=0, x2>=0)
    new_format_problem_from_your_drawing = {
        "objective": "max", 
        "coeffs": [3, 2], 
        "constraints": [
            {"name": "(1) x1+2x2<=6", "lhs": [1, 2], "op": "<=", "rhs": 6},
            {"name": "(2) -x1+x2<=1", "lhs": [-1, 1], "op": "<=", "rhs": 1},
            {"name": "(3) x1 >= 0", "lhs": [1, 0], "op": ">=", "rhs": 0},
            {"name": "(4) x2 >= 0", "lhs": [0, 1], "op": ">=", "rhs": 0}
        ]
    }
    print("\n\n--- BẮT ĐẦU GIẢI BẰNG PHƯƠNG PHÁP HÌNH HỌC (BÀI TOÁN TỪ HÌNH VẼ TAY - ĐỊNH DẠNG MỚI) ---")
    solution_draw, logs_draw = solve_with_geometric_method(new_format_problem_from_your_drawing)
    print("\n--- LOG CHI TIẾT (DRAWING PROBLEM) ---\n")
    for log_entry in logs_draw: print(log_entry)
    print("\n--- KẾT QUẢ CUỐI CÙNG (DRAWING PROBLEM) ---\n")
    if solution_draw:
        plot_image_data_draw = solution_draw.pop("plot_image_base64", None)
        print(json.dumps(solution_draw, indent=2))
        if plot_image_data_draw:
            print("\n--- XỬ LÝ HÌNH ẢNH (DRAWING PROBLEM) ---")
            try:
                output_dir = "test_outputs"; os.makedirs(output_dir, exist_ok=True)
                header, encoded = plot_image_data_draw.split(",", 1)
                image_data = base64.b64decode(encoded)
                output_path = os.path.join(output_dir, "geometric_solver_drawing_new_format.png") 
                with open(output_path, "wb") as f: f.write(image_data)
                print(f"[INFO] Đã lưu hình ảnh đồ thị vào: {output_path}")
                webbrowser.open(os.path.realpath(output_path))
            except Exception as e: print(f"[ERROR] Không thể xử lý ảnh: {e}")

