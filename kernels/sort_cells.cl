//Largely inspired by https://github.com/bneukom/gpu-nbody/blob/master/kernels/nbody/buildtree.cl
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#ifdef HAS__debug
    #define DEBUG(id,x,val) _debug[id*8+x] = val
#else
    #define DEBUG(id,x,val) do {} while (0)
#endif

#define MAX_LOOP (10000)
kernel void sort_cells( 
#ifdef SINGLE_WORKGROUP
                 int num_worksets, 
                 global const int *worksets_begins, 
                 global const int *worksets_lengths, 
#else
                 int worksets_begin, 
                 int worksets_length, 
#endif
                 int _pos_length, 
                 global float * _pos ,
                 int _cell_pos_length, 
                 global float * _cell_pos ,
                 int _mass_length, 
                 global float * _mass ,
                 int _cell_mass_length, 
                 global float * _cell_mass ,
                 int _cell_tree_length, 
                 global int * _cell_tree ,
                 int _cell_body_count_length, 
                 global int * _cell_body_count ,
#ifdef HAS__cell_start
                 int _cell_start_length, 
                 global int * _cell_start ,
#endif
#ifdef HAS__cell_sorted
                 int _cell_sorted_length, 
                 global int * _cell_sorted ,
#endif
                 int _g_acc_length, 
                 global float * _g_acc ,
                 int _bottom_length, 
                 global int * _bottom ,
                 int _radius_length, 
                 global float * _radius ,
#ifdef HAS__debug
                 int _debug_length, 
                 global float * _debug ,
#endif
#ifdef HAS__octree_depth
                 int _octree_depth_length, 
                 global int * _octree_depth ,
#endif
#ifdef HAS__octree_pos
                 int _octree_pos_length, 
                 global float * _octree_pos ,
#endif
#ifdef HAS__preview_id
                 int _preview_id_length, 
                 global int * _preview_id ,
#endif
#ifdef HAS__preview_switch
                 int _preview_switch_length, 
                 global int * _preview_switch ,
#endif
                 int _max_depth_length, 
                 global int * _max_depth ,
                 int _incr_length, 
                 global int * _incr ,
                 int _theta_length, 
                 global float * _theta ,
                 int _softening_length, 
                 global float * _softening ,
                 int _world_scale_length, 
                 global float * _world_scale,
                 int _mass_scale_length, 
                 global float * _mass_scale,
                 int _g_temp_length, 
                 global float * _g_temp 
)
{
#ifdef HAS__cell_sorted
#ifndef SINGLE_WORKGROUP
    if(worksets_begin!=0) return;
#endif

    const int NUM_BODIES = _cell_pos_length-1;
    const int NUM_NODES = NUM_BODIES*2+1;
    const int CELL_OFFSET = NUM_BODIES+1;
    const int NUM_CELLS = 8;
    

    int bottom = *_bottom;
    int step_size = get_local_size(0)*get_num_groups(0);
    int idx = get_global_id(0);


    int cell_idx = NUM_BODIES + 1 - step_size + idx;
    int i=0;
    while(cell_idx >= bottom && i < MAX_LOOP)
    {
        int start = _cell_start[cell_idx];
        if(start>=0)
        {
            _debug[cell_idx*8+3] += 1;
            #pragma unroll 8
            for(int j = 0; j < NUM_CELLS; ++j)
            {
                int child = _cell_tree[cell_idx * NUM_CELLS + j];
                //child is a cell
                if(child > NUM_BODIES)
                {
                    _cell_start[child-CELL_OFFSET] = start;

                    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
                    start += _cell_body_count[child-CELL_OFFSET];
                }
                else if(child >= 0)
                {
                    //Never saves the last body - FIX
                    _cell_sorted[start] = child;
                    ++start;
                }
            }
            cell_idx -= step_size;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        i++;
    }
#endif
}