#include "BiattSemValue.h"

BiattSemValue::BiattSemValue(bool is_tree){
	this->src_tree = NULL;
	this->tgt_tree = NULL;
	this->is_tree = is_tree;
	this->semScore = -0.0;
	this->theta_src_att_mat = NULL;
	this->theta_tgt_att_mat = NULL;
	this->grand_src_att_mat = NULL;
	this->grand_tgt_att_mat = NULL;
	this->grand_src_att_score = NULL;
	this->grand_tgt_att_score = NULL;
	this->theta_src_att_score = NULL;
	this->theta_tgt_att_score = NULL;
	this->theta_src_rae_mat = NULL;
	this->theta_tgt_rae_mat = NULL;
	this->grand_src_sem_rep = NULL;
	this->grand_tgt_sem_rep = NULL;
	this->theta_src_sem_rep = NULL;
	this->theta_tgt_sem_rep = NULL;
	this->theta_src_inst_representation = NULL;
	this->theta_tgt_inst_representation = NULL;
	this->biattention_matrix = NULL;
	this->grand_biattention_matrix = NULL;
	this->grand_v_score = NULL;
}
BiattSemValue::~BiattSemValue(){
	if(this->src_tree != NULL) delete this->src_tree;
	if(this->tgt_tree != NULL) delete this->tgt_tree;
	if(this->theta_src_att_mat != NULL) __LBFGSFREE__(this->theta_src_att_mat);
	if(this->theta_tgt_att_mat != NULL) __LBFGSFREE__(this->theta_tgt_att_mat);
	if(this->grand_src_att_mat != NULL) __LBFGSFREE__(this->grand_src_att_mat);
	if(this->grand_tgt_att_mat != NULL) __LBFGSFREE__(this->grand_tgt_att_mat);
	if(this->grand_src_att_score != NULL) __LBFGSFREE__(this->grand_src_att_score);
	if(this->grand_tgt_att_score != NULL) __LBFGSFREE__(this->grand_tgt_att_score);
	if(this->theta_src_att_score != NULL) __LBFGSFREE__(this->theta_src_att_score);
	if(this->theta_tgt_att_score != NULL) __LBFGSFREE__(this->theta_tgt_att_score);
	if(this->theta_src_rae_mat != NULL) __LBFGSFREE__(this->theta_src_rae_mat);
	if(this->theta_tgt_rae_mat != NULL) __LBFGSFREE__(this->theta_tgt_rae_mat);
	if(this->grand_src_sem_rep != NULL) __LBFGSFREE__(this->grand_src_sem_rep);
	if(this->grand_tgt_sem_rep != NULL) __LBFGSFREE__(this->grand_tgt_sem_rep);
	if(this->theta_src_sem_rep != NULL) __LBFGSFREE__(this->theta_src_sem_rep);
	if(this->theta_tgt_sem_rep != NULL) __LBFGSFREE__(this->theta_tgt_sem_rep);
	if(this->theta_src_inst_representation != NULL) __LBFGSFREE__(this->theta_src_inst_representation);
	if(this->theta_tgt_inst_representation != NULL) __LBFGSFREE__(this->theta_tgt_inst_representation);
	if(this->grand_v_score != NULL) __LBFGSFREE__(this->grand_v_score);
	if(this->biattention_matrix != NULL) __LBFGSFREE__(this->biattention_matrix);
	if(this->grand_biattention_matrix != NULL) __LBFGSFREE__(this->grand_biattention_matrix);
}
