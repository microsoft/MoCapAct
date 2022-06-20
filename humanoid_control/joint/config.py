from ml_collections import ConfigDict
from humanoid_control import observables

def prepend_model(name):
    return "humanoid_control.joint.model." + name

def get_config(config_string):
    configs = {
        'mlp': ConfigDict({
            'constructor': "stable_baselines3.common.policies.MultiInputActorCriticPolicy",
            'config': ConfigDict(dict(
                n_layers=3,
                layer_size=1024,
                activation_fn='torch.nn.Tanh',
                squash_output=False,
                observables=observables.BASE_OBSERVABLES
            ))
        }),

        'hierarchical': ConfigDict({
            'constructor': prepend_model('HierarchicalPolicy'),
            'config': ConfigDict(dict(
                embed_size=60,
                ref_encoder_n_layers=2,
                ref_encoder_layer_size=1024,
                stand_encoder_n_layers=2,
                stand_encoder_layer_size=1024,
                decoder_n_layers=3,
                decoder_layer_size=1024,
                stand_vf_n_layers=3,
                stand_vf_layer_size=1024,
                layer_norm=True,
                embedding_kl_weight=0.1,
                embedding_correlation=0.95,
                seq_steps=30,
                activation_fn='torch.nn.Tanh',
                squash_output=False,
                observables=observables.HYBRID_HIERARCHICAL_OBSERVABLES
            ))
        }),

        'stand': ConfigDict({
            'constructor': prepend_model('StandUpPolicy'),
            'config': ConfigDict(dict(
                embed_size=60,
                ref_encoder_n_layers=2,
                ref_encoder_layer_size=1024,
                decoder_n_layers=3,
                decoder_layer_size=1024,
                stand_vf_n_layers=3,
                stand_vf_layer_size=1024,
                layer_norm=True,
                ref_embedding_kl_weight=0.1,
                embedding_correlation=0.95,
                bc_seq_steps=30,
                activation_fn='torch.nn.Tanh',
                squash_output=False,
                observables=observables.SIMPLE_STAND_UP_OBSERVABLES
            ))
        }),


        'hierarchical_clip': ConfigDict({
            'constructor': prepend_model('HierarchicalPolicy'),
            'config': ConfigDict(dict(
                embed_size=20,
                ref_encoder_n_layers=2,
                ref_encoder_layer_size=1024,
                stand_encoder_n_layers=2,
                stand_encoder_layer_size=1024,
                decoder_n_layers=3,
                decoder_layer_size=1024,
                stand_vf_n_layers=3,
                stand_vf_layer_size=1024,
                layer_norm=True,
                embedding_kl_weight=0.1,
                embedding_correlation=0.95,
                seq_steps=30,
                activation_fn='torch.nn.Tanh',
                squash_output=False,
                observables=observables.CLIP_HIERARCHICAL_OBSERVABLES
            ))
        })
    }

    return configs[config_string]
