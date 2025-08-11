"""
Model Evaluator - Performance analysis and comparison
"""
import time
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.data_manager import DataManager, QAPair

class ModelEvaluator:
    """Evaluates student model performance and provides analysis"""
    
    def __init__(self, config):
        self.config = config
        self.evaluation_results = []
        self.baseline_results = None
        self.comparison_results = {}
    
    def run_baseline_evaluation(self, student_ai, teacher_ai, 
                              test_data: List[QAPair]) -> Dict[str, Any]:
        """Run baseline evaluation on test dataset"""
        print(f"Running baseline evaluation with {len(test_data)} questions...")
        
        start_time = time.time()
        results = []
        
        for i, qa_pair in enumerate(test_data):
            print(f"Evaluating question {i+1}/{len(test_data)}: {qa_pair.question[:50]}...")
            
            # Get student response
            student_answer = student_ai.generate_response(qa_pair.question)
            
            # Get teacher evaluation
            evaluation = teacher_ai.evaluate_answer(
                question=qa_pair.question,
                correct_answer=qa_pair.answer,
                student_answer=student_answer
            )
            
            result = {
                "question_id": i,
                "question": qa_pair.question,
                "expected_answer": qa_pair.answer,
                "student_answer": student_answer,
                "score": evaluation.get("score", 0),
                "feedback": evaluation.get("feedback", ""),
                "timestamp": time.time()
            }
            
            results.append(result)
            
            # Small delay to prevent API overload
            time.sleep(0.5)
        
        end_time = time.time()
        
        # Calculate statistics
        scores = [r["score"] for r in results]
        
        baseline_stats = {
            "total_questions": len(results),
            "average_score": statistics.mean(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0,
            "evaluation_duration": end_time - start_time,
            "score_distribution": self._calculate_score_distribution(scores)
        }
        
        self.baseline_results = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "statistics": baseline_stats,
            "evaluation_type": "baseline"
        }
        
        print(f"Baseline evaluation completed!")
        print(f"Average score: {baseline_stats['average_score']:.2f}/10")
        print(f"Score range: {baseline_stats['min_score']}-{baseline_stats['max_score']}")
        print(f"Duration: {baseline_stats['evaluation_duration']:.2f} seconds")
        
        return self.baseline_results
    
    def run_post_training_evaluation(self, student_ai, teacher_ai, 
                                   test_data: List[QAPair], 
                                   training_session_id: str) -> Dict[str, Any]:
        """Run evaluation after training to measure improvement"""
        print(f"Running post-training evaluation...")
        
        start_time = time.time()
        results = []
        
        for i, qa_pair in enumerate(test_data):
            print(f"Evaluating question {i+1}/{len(test_data)}: {qa_pair.question[:50]}...")
            
            student_answer = student_ai.generate_response(qa_pair.question)
            
            evaluation = teacher_ai.evaluate_answer(
                question=qa_pair.question,
                correct_answer=qa_pair.answer,
                student_answer=student_answer
            )
            
            result = {
                "question_id": i,
                "question": qa_pair.question,
                "expected_answer": qa_pair.answer,
                "student_answer": student_answer,
                "score": evaluation.get("score", 0),
                "feedback": evaluation.get("feedback", ""),
                "timestamp": time.time(),
                "training_session": training_session_id
            }
            
            results.append(result)
            time.sleep(0.5)
        
        end_time = time.time()
        
        # Calculate statistics
        scores = [r["score"] for r in results]
        
        post_training_stats = {
            "total_questions": len(results),
            "average_score": statistics.mean(scores) if scores else 0,
            "median_score": statistics.median(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0,
            "evaluation_duration": end_time - start_time,
            "score_distribution": self._calculate_score_distribution(scores)
        }
        
        post_training_results = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "statistics": post_training_stats,
            "evaluation_type": "post_training",
            "training_session_id": training_session_id
        }
        
        # Compare with baseline if available
        if self.baseline_results:
            comparison = self._compare_evaluations(
                self.baseline_results["statistics"], 
                post_training_stats
            )
            post_training_results["comparison_with_baseline"] = comparison
            
            print(f"\n=== Improvement Analysis ===")
            print(f"Score improvement: {comparison['score_improvement']:+.2f} points")
            print(f"Relative improvement: {comparison['relative_improvement']:+.2f}%")
        
        print(f"Post-training evaluation completed!")
        print(f"Average score: {post_training_stats['average_score']:.2f}/10")
        
        return post_training_results
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of scores across ranges"""
        distribution = {
            "excellent (9-10)": 0,
            "good (7-8)": 0,
            "fair (5-6)": 0,
            "poor (0-4)": 0
        }
        
        for score in scores:
            if 9 <= score <= 10:
                distribution["excellent (9-10)"] += 1
            elif 7 <= score < 9:
                distribution["good (7-8)"] += 1
            elif 5 <= score < 7:
                distribution["fair (5-6)"] += 1
            else:
                distribution["poor (0-4)"] += 1
        
        return distribution
    
    def _compare_evaluations(self, baseline_stats: Dict, 
                           post_training_stats: Dict) -> Dict[str, Any]:
        """Compare baseline and post-training evaluation results"""
        baseline_avg = baseline_stats.get("average_score", 0)
        post_training_avg = post_training_stats.get("average_score", 0)
        
        score_improvement = post_training_avg - baseline_avg
        relative_improvement = (score_improvement / baseline_avg * 100) if baseline_avg > 0 else 0
        
        return {
            "baseline_average": baseline_avg,
            "post_training_average": post_training_avg,
            "score_improvement": score_improvement,
            "relative_improvement": relative_improvement,
            "improvement_significant": abs(score_improvement) > 0.5,
            "baseline_std": baseline_stats.get("std_deviation", 0),
            "post_training_std": post_training_stats.get("std_deviation", 0)
        }
    
    def generate_evaluation_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AI TRAINING SYSTEM - EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Baseline Results
        if self.baseline_results:
            report_lines.append("BASELINE EVALUATION")
            report_lines.append("-" * 30)
            stats = self.baseline_results["statistics"]
            report_lines.append(f"Total Questions: {stats['total_questions']}")
            report_lines.append(f"Average Score: {stats['average_score']:.2f}/10")
            report_lines.append(f"Median Score: {stats['median_score']:.2f}/10")
            report_lines.append(f"Score Range: {stats['min_score']:.1f} - {stats['max_score']:.1f}")
            report_lines.append(f"Standard Deviation: {stats['std_deviation']:.2f}")
            report_lines.append("")
            
            report_lines.append("Score Distribution:")
            for category, count in stats['score_distribution'].items():
                percentage = (count / stats['total_questions'] * 100)
                report_lines.append(f"  {category}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # Training Improvements
        if self.comparison_results:
            report_lines.append("TRAINING IMPROVEMENTS")
            report_lines.append("-" * 30)
            for session_id, comparison in self.comparison_results.items():
                report_lines.append(f"Session: {session_id}")
                report_lines.append(f"  Score Improvement: {comparison['score_improvement']:+.2f} points")
                report_lines.append(f"  Relative Improvement: {comparison['relative_improvement']:+.2f}%")
                report_lines.append(f"  Significant: {'Yes' if comparison['improvement_significant'] else 'No'}")
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Evaluation report saved to: {output_file}")
        
        return report
    
    def create_performance_chart(self, output_file: str = "performance_chart.png"):
        """Create performance visualization chart"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('AI Training System - Performance Analysis', fontsize=16, fontweight='bold')
            
            # Chart 1: Score Distribution (Baseline vs Post-Training)
            if self.baseline_results:
                baseline_dist = self.baseline_results["statistics"]["score_distribution"]
                categories = list(baseline_dist.keys())
                baseline_values = list(baseline_dist.values())
                
                axes[0, 0].bar(categories, baseline_values, alpha=0.7, label='Baseline')
                axes[0, 0].set_title('Score Distribution Comparison')
                axes[0, 0].set_ylabel('Number of Questions')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].legend()
            
            # Chart 2: Score Improvement Over Time
            if self.comparison_results:
                sessions = list(self.comparison_results.keys())
                improvements = [comp['score_improvement'] for comp in self.comparison_results.values()]
                
                axes[0, 1].plot(range(len(sessions)), improvements, marker='o', linewidth=2)
                axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0, 1].set_title('Score Improvement Over Sessions')
                axes[0, 1].set_xlabel('Training Session')
                axes[0, 1].set_ylabel('Score Improvement')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Chart 3: Average Scores Comparison
            if self.baseline_results and self.comparison_results:
                baseline_avg = self.baseline_results["statistics"]["average_score"]
                post_training_avgs = [comp['post_training_average'] for comp in self.comparison_results.values()]
                
                x_pos = [0] + list(range(1, len(post_training_avgs) + 1))
                y_values = [baseline_avg] + post_training_avgs
                colors = ['red'] + ['green'] * len(post_training_avgs)
                
                axes[1, 0].bar(x_pos, y_values, color=colors, alpha=0.7)
                axes[1, 0].set_title('Average Scores: Baseline vs Post-Training')
                axes[1, 0].set_xlabel('Evaluation')
                axes[1, 0].set_ylabel('Average Score')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(['Baseline'] + [f'Session {i+1}' for i in range(len(post_training_avgs))])
            
            # Chart 4: Performance Metrics Summary
            if self.baseline_results:
                metrics = ['Average', 'Median', 'Min', 'Max']
                stats = self.baseline_results["statistics"]
                values = [stats['average_score'], stats['median_score'], stats['min_score'], stats['max_score']]
                
                axes[1, 1].bar(metrics, values, color='skyblue', alpha=0.7)
                axes[1, 1].set_title('Performance Metrics Summary')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_ylim(0, 10)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Performance chart saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating performance chart: {e}")
            return None
    
    def run_comprehensive_evaluation(self, student_ai, teacher_ai, 
                                   test_data: List[QAPair],
                                   training_session_id: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation with all analyses"""
        print("Starting comprehensive evaluation...")
        
        # Run baseline if not done yet
        if not self.baseline_results:
            self.run_baseline_evaluation(student_ai, teacher_ai, test_data)
        
        # Run post-training evaluation if training session provided
        post_training_results = None
        if training_session_id:
            post_training_results = self.run_post_training_evaluation(
                student_ai, teacher_ai, test_data, training_session_id
            )
        
        # Generate report and charts
        report_file = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        chart_file = f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        report = self.generate_evaluation_report(report_file)
        chart = self.create_performance_chart(chart_file)
        
        return {
            "baseline_results": self.baseline_results,
            "post_training_results": post_training_results,
            "comparison_results": self.comparison_results,
            "report_file": report_file,
            "chart_file": chart,
            "report_text": report
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed"""
        summary = {
            "baseline_completed": self.baseline_results is not None,
            "training_sessions_evaluated": len(self.comparison_results),
            "evaluation_history": []
        }
        
        if self.baseline_results:
            summary["baseline_stats"] = self.baseline_results["statistics"]
        
        for session_id, comparison in self.comparison_results.items():
            summary["evaluation_history"].append({
                "session_id": session_id,
                "improvement": comparison["score_improvement"],
                "significant": comparison["improvement_significant"]
            })
        
        return summary